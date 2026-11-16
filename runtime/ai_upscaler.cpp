#include "ai_upscaler.h"

#include <d3dcompiler.h>

#include <cstdio>
#include <cstring>

#include "png.h"

using Microsoft::WRL::ComPtr;

namespace nax {

namespace {

// Each convolution is one dispatch. A thread computes 16 output channels of one pixel from the 3x3
// neighbourhood of all input channels, with zero padding at the picture border as in the original network.
// Feature maps are stored channel by channel, so neighbouring threads read neighbouring values.
// The last step rearranges the 48 output channels into a picture four times as large ("pixel shuffle") and
// adds the original picture, enlarged without smoothing, as the network expects.
const char kShader[] = R"(
cbuffer Layer : register(b0) { uint w_off; uint b_off; uint a_off; uint cin; uint flags; uint unused; };
Texture2D<float4> picture : register(t0);
StructuredBuffer<float> weights : register(t1);
RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<float> fout : register(u1);
RWTexture2D<float4> result : register(u2);
static const int W = 320;
static const int H = 200;
static const int N = W * H;
#define OUTS 16

float input_at(uint c, int x, int y) {
    if (x < 0 || y < 0 || x >= W || y >= H) return 0;
    if (flags & 1) return picture.Load(int3(x, y, 0))[c];
    return fin[c * N + y * W + x];
}

[numthreads(16, 8, 1)]
void conv(uint3 id : SV_DispatchThreadID) {
    int x = id.x, y = id.y;
    if (x >= W || y >= H) return;
    uint ob = id.z * OUTS;
    float acc[OUTS];
    [unroll] for (uint k = 0; k < OUTS; k++) acc[k] = weights[b_off + ob + k];
    for (uint c = 0; c < cin; c++) {
        float v[9];
        [unroll] for (int j = 0; j < 9; j++) v[j] = input_at(c, x + j % 3 - 1, y + j / 3 - 1);
        [unroll] for (uint k = 0; k < OUTS; k++) {
            uint base = w_off + ((ob + k) * cin + c) * 9;
            float s = 0;
            [unroll] for (int t = 0; t < 9; t++) s += weights[base + t] * v[t];
            acc[k] += s;
        }
    }
    [unroll] for (uint k = 0; k < OUTS; k++) {
        float a = acc[k];
        if ((flags & 2) && a < 0) a *= weights[a_off + ob + k];
        fout[(ob + k) * N + y * W + x] = a;
    }
}

[numthreads(16, 8, 1)]
void shuffle(uint3 id : SV_DispatchThreadID) {
    int X = id.x, Y = id.y;
    if (X >= W * 4 || Y >= H * 4) return;
    int x = X / 4, y = Y / 4, i = Y % 4, j = X % 4;
    float4 base = picture.Load(int3(x, y, 0));
    float r = fin[(0 * 16 + i * 4 + j) * N + y * W + x] + base.r;
    float g = fin[(1 * 16 + i * 4 + j) * N + y * W + x] + base.g;
    float b = fin[(2 * 16 + i * 4 + j) * N + y * W + x] + base.b;
    result[int2(X, Y)] = float4(saturate(float3(r, g, b)), 1);
}
)";

constexpr UINT kOuts = 16;

bool fail(const char* what, HRESULT hr, std::string* error) {
    char text[160];
    std::snprintf(text, sizeof(text), "%s failed (0x%08lX).", what, static_cast<unsigned long>(hr));
    *error = text;
    return false;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return b;
}

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    ComPtr<ID3D12Resource> r;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

}  // namespace

bool AiUpscaler::init(ID3D12Device* device, UINT64 timestamp_frequency, const std::filesystem::path& path, std::string* error) {
    device_ = device;
    timestamp_frequency_ = timestamp_frequency;
    FILE* f = _wfopen(path.c_str(), L"rb");
    if (!f) { *error = "The AI model file was not found."; return false; }
    std::vector<uint8_t> raw;
    uint8_t chunk[65536];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) raw.insert(raw.end(), chunk, chunk + n);
    std::fclose(f);
    if (raw.size() < 20 || std::memcmp(raw.data(), "NAXSRV1\0", 8) != 0) { *error = "The AI model file has an unknown format."; return false; }
    uint32_t header[3];
    std::memcpy(header, raw.data() + 8, sizeof(header));
    const uint32_t count = header[0], features = header[1], scale = header[2];
    if (scale != kScale || features % kOuts != 0 || count < 2) { *error = "The AI model does not match this filter."; return false; }

    // The file holds the per-layer counts followed by the floats; the floats go to the GPU in the same order.
    std::vector<float> floats;
    size_t pos = 20;
    for (uint32_t i = 0; i < count; i++) {
        if (pos + 8 > raw.size()) { *error = "The AI model file is truncated."; return false; }
        uint32_t io[2];
        std::memcpy(io, raw.data() + pos, 8);
        pos += 8;
        Layer layer{};
        layer.in_channels = io[0];
        layer.out_channels = io[1];
        layer.prelu = i + 1 < count;
        const bool shape_ok = layer.out_channels % kOuts == 0 && (i == 0 ? layer.in_channels == 3 : layer.in_channels == features) &&
                              (i + 1 < count ? layer.out_channels == features : layer.out_channels == 3 * kScale * kScale);
        if (!shape_ok) { *error = "The AI model has an unexpected layer shape."; return false; }
        const size_t values = size_t(layer.out_channels) * layer.in_channels * 9 + layer.out_channels + (layer.prelu ? layer.out_channels : 0);
        if (pos + values * 4 > raw.size()) { *error = "The AI model file is truncated."; return false; }
        layer.weights = uint32_t(floats.size());
        layer.biases = layer.weights + layer.out_channels * layer.in_channels * 9;
        layer.slopes = layer.biases + layer.out_channels;
        const size_t start = floats.size();
        floats.resize(start + values);
        std::memcpy(&floats[start], raw.data() + pos, values * 4);
        pos += values * 4;
        layers_.push_back(layer);
    }

    weights_bytes_ = floats.size() * 4;
    weights_ = buffer(device, D3D12_HEAP_TYPE_DEFAULT, weights_bytes_, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    weights_upload_ = buffer(device, D3D12_HEAP_TYPE_UPLOAD, weights_bytes_, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    const UINT64 feature_bytes = UINT64(features) * kInWidth * kInHeight * 4;
    for (auto& fb : features_)
        fb = buffer(device, D3D12_HEAP_TYPE_DEFAULT, feature_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!weights_ || !weights_upload_ || !features_[0] || !features_[1]) { *error = "Creating the AI buffers failed."; return false; }
    void* mapped = nullptr;
    D3D12_RANGE none{0, 0};
    if (FAILED(weights_upload_->Map(0, &none, &mapped))) { *error = "Mapping the AI weights failed."; return false; }
    std::memcpy(mapped, floats.data(), weights_bytes_);
    weights_upload_->Unmap(0, nullptr);

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = kOutWidth;
    td.Height = kOutHeight;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&result_));
    if (FAILED(hr)) return fail("Creating the AI result texture", hr, error);

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 2;  // the game picture, then the result
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_));
    if (FAILED(hr)) return fail("Creating the AI descriptor heap", hr, error);
    auto slot = heap_->GetCPUDescriptorHandleForHeapStart();
    slot.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(result_.Get(), nullptr, &uav, slot);

    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 2;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER params[5]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 6;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 1;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 1;
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[4].DescriptorTable.NumDescriptorRanges = 2;
    params[4].DescriptorTable.pDescriptorRanges = ranges;
    D3D12_ROOT_SIGNATURE_DESC rd{};
    rd.NumParameters = 5;
    rd.pParameters = params;
    ComPtr<ID3DBlob> blob, errors;
    hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors);
    if (FAILED(hr)) return fail("Building the AI root signature", hr, error);
    hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_));
    if (FAILED(hr)) return fail("Creating the AI root signature", hr, error);

    using Entry = std::pair<const char*, ID3D12PipelineState**>;
    for (auto [entry, pso] : {Entry{"conv", conv_.GetAddressOf()}, Entry{"shuffle", shuffle_.GetAddressOf()}}) {
        ComPtr<ID3DBlob> code;
        errors.Reset();
        hr = D3DCompile(kShader, sizeof(kShader) - 1, "ai", nullptr, nullptr, entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(hr)) {
            fail("Compiling the AI shader", hr, error);
            if (errors) *error += std::string(" ") + static_cast<const char*>(errors->GetBufferPointer());
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC cd{};
        cd.pRootSignature = root_.Get();
        cd.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = device->CreateComputePipelineState(&cd, IID_PPV_ARGS(pso));
        if (FAILED(hr)) return fail("Creating the AI pipeline", hr, error);
    }
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = 2;
    if (FAILED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&queries_)))) queries_.Reset();
    query_results_ = buffer(device, D3D12_HEAP_TYPE_READBACK, 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ready_ = true;
    return true;
}

double AiUpscaler::last_gpu_ms() {
    if (!timed_ || !query_results_ || timestamp_frequency_ == 0) return -1;
    uint64_t* t = nullptr;
    D3D12_RANGE range{0, 16};
    if (FAILED(query_results_->Map(0, &range, reinterpret_cast<void**>(&t)))) return -1;
    const double ms = double(t[1] - t[0]) * 1000.0 / double(timestamp_frequency_);
    D3D12_RANGE none{0, 0};
    query_results_->Unmap(0, &none);
    return ms;
}

void AiUpscaler::record(ID3D12GraphicsCommandList* list, ID3D12Resource* picture) {
    if (!ready_) return;
    if (!weights_copied_) {
        list->CopyBufferRegion(weights_.Get(), 0, weights_upload_.Get(), 0, weights_bytes_);
        const auto b = transition(weights_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &b);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(picture, &srv, heap_->GetCPUDescriptorHandleForHeapStart());
        weights_copied_ = true;
    }
    if (result_readable_) {
        const auto b = transition(result_.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &b);
    }
    if (queries_) list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    list->SetComputeRootSignature(root_.Get());
    ID3D12DescriptorHeap* heaps[] = {heap_.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootDescriptorTable(4, heap_->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRootShaderResourceView(1, weights_->GetGPUVirtualAddress());
    list->SetPipelineState(conv_.Get());
    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    int src = 0;
    for (size_t i = 0; i < layers_.size(); i++) {
        const Layer& l = layers_[i];
        const uint32_t constants[6] = {l.weights, l.biases, l.slopes, l.in_channels, (i == 0 ? 1u : 0u) | (l.prelu ? 2u : 0u), 0};
        list->SetComputeRoot32BitConstants(0, 6, constants, 0);
        list->SetComputeRootUnorderedAccessView(2, features_[src]->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(3, features_[1 - src]->GetGPUVirtualAddress());
        list->Dispatch((kInWidth + 15) / 16, (kInHeight + 7) / 8, l.out_channels / kOuts);
        list->ResourceBarrier(1, &uav_barrier);
        src = 1 - src;
    }
    list->SetPipelineState(shuffle_.Get());
    list->SetComputeRootUnorderedAccessView(2, features_[src]->GetGPUVirtualAddress());
    list->SetComputeRootUnorderedAccessView(3, features_[1 - src]->GetGPUVirtualAddress());
    list->Dispatch(kOutWidth / 16, kOutHeight / 8, 1);
    const auto done = transition(result_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &done);
    result_readable_ = true;
    if (queries_ && query_results_) {
        list->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        list->ResolveQueryData(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, query_results_.Get(), 0);
        timed_ = true;
    }
}

void AiUpscaler::record_readback(ID3D12GraphicsCommandList* list) {
    if (!ready_ || !result_readable_) return;
    const D3D12_RESOURCE_DESC desc = result_->GetDesc();
    UINT64 total = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &readback_footprint_, nullptr, nullptr, &total);
    if (!readback_) readback_ = buffer(device_.Get(), D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!readback_) return;
    const auto to_copy = transition(result_.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION from{};
    from.pResource = result_.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION into{};
    into.pResource = readback_.Get();
    into.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    into.PlacedFootprint = readback_footprint_;
    list->CopyTextureRegion(&into, 0, 0, 0, &from, nullptr);
    const auto back = transition(result_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list->ResourceBarrier(1, &back);
}

bool AiUpscaler::save_readback(const std::filesystem::path& path) {
    if (!readback_) return false;
    uint8_t* data = nullptr;
    D3D12_RANGE range{0, SIZE_T(readback_footprint_.Footprint.RowPitch) * kOutHeight};
    if (FAILED(readback_->Map(0, &range, reinterpret_cast<void**>(&data)))) return false;
    std::vector<uint32_t> pixels(size_t(kOutWidth) * kOutHeight);
    for (UINT y = 0; y < kOutHeight; y++)
        std::memcpy(&pixels[size_t(y) * kOutWidth], data + readback_footprint_.Offset + size_t(y) * readback_footprint_.Footprint.RowPitch, kOutWidth * 4);
    D3D12_RANGE none{0, 0};
    readback_->Unmap(0, &none);
    return write_png(path, pixels.data(), kOutWidth, kOutHeight, 1);
}

}  // namespace nax
