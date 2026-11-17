#include "presenter.h"

#include <d3dcompiler.h>
#include <wincodec.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "png.h"

using Microsoft::WRL::ComPtr;

namespace nax {

namespace {

// A full-screen triangle draws the 320x200 picture with one of three filters, chosen by "mode":
//   0  Sharp pixels: each source pixel is enlarged by the largest whole factor and only the remaining fraction
//      is blended ("sharp bilinear"), which keeps pixels crisp at any window size.
//   1  Smooth edges: the EPX rule (also known as Scale2x) is applied twice, giving a picture four times as large
//      in which stair steps along diagonal and round edges become smooth lines while flat areas stay unchanged.
//   2  CRT: the picture as a colour monitor of the time showed it, with visible scanlines, a slight glow
//      between them and an aperture grille, computed in linear light.
//   3  AI upscaling: where a sprite has an upscaled picture (see hd_sprites.h), that picture is drawn at the
//      full screen resolution. Everything else is redrawn by a neural network (see ai_upscaler.cpp) at four
//      times the size, and every screen pixel is then filled with the closest of the game's own colours.
const char kShader[] = R"(
cbuffer Params : register(b0) { float2 scale; uint mode; uint unused; };
Texture2D picture : register(t0);
Texture2D upscaled : register(t1);
Texture2D atlas : register(t2);
Texture2D<uint> refs : register(t3);
SamplerState smooth : register(s0);
static const float2 size = float2(320, 200);
struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
V vs(uint id : SV_VertexID) {
    V o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float3 sharp_of(Texture2D t, float2 tsize, float2 tscale, float2 uv) {
    float2 texel = uv * tsize;
    float2 whole = max(floor(tscale), 1);
    float2 base = floor(texel);
    float2 dist = frac(texel) - 0.5;
    float2 range = 0.5 - 0.5 / whole;
    float2 f = (dist - clamp(dist, -range, range)) * whole + 0.5;
    return t.SampleLevel(smooth, (base + f) / tsize, 0).rgb;
}
float3 sharp(float2 uv) { return sharp_of(picture, size, scale, uv); }

float3 texel(int2 p) { return picture.Load(int3(clamp(p, int2(0, 0), int2(319, 199)), 0)).rgb; }
bool same(float3 a, float3 b) { return all(abs(a - b) < 0.002); }

// One EPX step: the quarter q of pixel P, from its neighbours above (A), right (B), left (C) and below (D).
float3 epx(float3 P, float3 A, float3 B, float3 C, float3 D, int2 q) {
    if (q.x == 0 && q.y == 0) return (same(C, A) && !same(C, D) && !same(A, B)) ? A : P;
    if (q.x == 1 && q.y == 0) return (same(A, B) && !same(A, C) && !same(B, D)) ? B : P;
    if (q.x == 0 && q.y == 1) return (same(D, C) && !same(D, B) && !same(C, A)) ? C : P;
    return (same(B, D) && !same(B, A) && !same(D, C)) ? D : P;
}
float3 double_size(int2 p2) {
    int2 p = p2 >> 1;
    return epx(texel(p), texel(p + int2(0, -1)), texel(p + int2(1, 0)), texel(p + int2(-1, 0)), texel(p + int2(0, 1)), p2 & 1);
}
float3 smooth_edges(float2 uv) {
    int2 p4 = clamp(int2(floor(uv * size * 4)), int2(0, 0), int2(1279, 799));
    int2 p2 = p4 >> 1;
    return epx(double_size(p2), double_size(p2 + int2(0, -1)), double_size(p2 + int2(1, 0)), double_size(p2 + int2(-1, 0)),
               double_size(p2 + int2(0, 1)), p4 & 1);
}

// The network's own picture is soft: it blurs letters and icons, erases fine textures such as soil speckle
// and turns background noise into streaks, and on a large screen it has to be enlarged further. So only the
// shapes it draws are used, never its colours. Each screen pixel takes the network's colour at that point and
// replaces it with the closest of the game's own colours around the source pixel, which gives hard, smooth
// edges at the full screen resolution. Only neighbours that differ strongly from the pixel (more than 0.3 in
// one channel) may take over its area, so the low-contrast textures keep their original pixels, and the
// centre of every pixel keeps its colour, so single dots such as cracks and windows never disappear.
float largest(float3 v) { return max(max(v.r, v.g), v.b); }
float3 snapped(float2 uv) {
    float3 network = upscaled.SampleLevel(smooth, uv, 0).rgb;
    float2 s = uv * size;
    int2 p = int2(floor(s));
    float3 own = texel(p);
    if (length(s - p - 0.5) < 0.3) return own;
    float3 best = own;
    float best_distance = length(network - own) * 1.2;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float3 c = texel(p + int2(dx, dy));
            float d = length(network - c);
            if (largest(abs(c - own)) > 0.3 && d < best_distance) {
                best = c;
                best_distance = d;
            }
        }
    }
    return best;
}
float3 ai_shapes(float2 uv) {
    // Four samples per screen pixel smooth the steps that the hard colour choice leaves along edges.
    float2 o = 0.25 / (size * scale);
    return (snapped(uv + float2(-o.x, -o.y)) + snapped(uv + float2(o.x, -o.y)) + snapped(uv + float2(-o.x, o.y)) +
            snapped(uv + float2(o.x, o.y))) * 0.25;
}

// A sprite's upscaled picture has a smooth outline, while the game's pixels have stair steps. So at every
// point the picture of the pixel underneath and those of its eight neighbours are read, each continued to this
// point, and of those whose outline covers the point, the one drawn last is shown. A neighbour belonging to the
// same sprite continues the same picture, so only other sprites and the ground compete.
float4 hd_part(uint r, float2 local) {
    float2 cell = float2(r & 0x3FF, (r >> 10) & 0x3FF);
    return atlas.SampleLevel(smooth, (cell + local) / 1024.0, 0);  // the atlas spans 1024 original pixels
}
int order_of(uint r) { return int(r >> 20); }
// True when drawing order a lies above b, with the count starting again after 4095 (0 is always below).
bool above(int a, int b) {
    if (b == 0) return a != 0;
    if (a == 0) return false;
    int d = (a - b + 4095) % 4095;
    return d != 0 && d < 2048;
}
float3 ai_picture(float2 uv) {
    float2 s = uv * size;
    int2 p = clamp(int2(floor(s)), int2(0, 0), int2(319, 199));
    float2 f = s - p;
    uint r = refs.Load(int3(p, 0));
    if (r == 0xFFFFFFFFu) return ai_shapes(uv);
    float4 own = hd_part(r, f);
    float3 colour = own.rgb;
    int top = own.a >= 0.5 ? order_of(r) : -1;
    [unroll] for (int dy = -1; dy <= 1; dy++) {
        [unroll] for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int2 q = p + int2(dx, dy);
            if (any(q < int2(0, 0)) || any(q > int2(319, 199))) continue;
            uint n = refs.Load(int3(q, 0));
            if (n == 0xFFFFFFFFu) continue;
            float4 other = hd_part(n, f - float2(dx, dy));
            if (other.a < 0.5) continue;
            int o = order_of(n);
            if (top < 0 || above(o, top)) {
                top = o;
                colour = other.rgb;
            }
        }
    }
    return colour;
}

float3 to_linear(float3 c) { return pow(max(c, 0), 2.4); }
float3 row_colour(float x, float row) {
    // Horizontally the beam blends neighbouring pixels a little; the sharp-bilinear rule keeps that slight.
    float whole = max(floor(scale.x), 1);
    float base = floor(x);
    float dist = frac(x) - 0.5;
    float range = 0.5 - 0.75 / whole;
    float f = (dist - clamp(dist, -range, range)) * whole + 0.5;
    return to_linear(picture.SampleLevel(smooth, float2((base + f) / size.x, (clamp(row, 0, 199) + 0.5) / size.y), 0).rgb);
}
float3 crt(float2 uv, float2 pos) {
    float2 src = uv * size;
    float y = src.y - 0.5;
    float row = floor(y);
    float f = y - row;
    float3 a = row_colour(src.x, row), b = row_colour(src.x, row + 1);
    // Each scanline is a beam with a Gaussian profile; bright lines are a little wider, as on a real tube.
    float3 wa = exp(-pow(f / (0.30 + 0.12 * a), 2) * 0.5);
    float3 wb = exp(-pow((1 - f) / (0.30 + 0.12 * b), 2) * 0.5);
    float3 c = a * wa + b * wb;
    if (scale.x >= 3) {
        uint m = uint(pos.x) % 3;
        c *= float3(m == 0 ? 1.20 : 0.80, m == 1 ? 1.20 : 0.80, m == 2 ? 1.20 : 0.80);
    }
    return pow(saturate(c * 1.35), 1 / 2.2);
}

float4 ps(V i) : SV_Target {
    if (mode == 1) return float4(smooth_edges(i.uv), 1);
    if (mode == 2) return float4(crt(i.uv, i.pos.xy), 1);
    if (mode == 3) return float4(ai_picture(i.uv), 1);
    return float4(sharp(i.uv), 1);
}
)";

bool check(HRESULT hr, const char* what, std::string* error) {
    if (SUCCEEDED(hr)) return true;
    char text[160];
    std::snprintf(text, sizeof(text), "%s failed (0x%08lX).", what, static_cast<unsigned long>(hr));
    *error = text;
    return false;
}

constexpr D3D12_RESOURCE_STATES kPictureReadable =
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return b;
}

}  // namespace

Presenter::~Presenter() {
    if (device_) wait_idle();
    if (fence_event_) CloseHandle(fence_event_);
}

bool Presenter::init(HWND hwnd, std::string* error) {
    hwnd_ = hwnd;
    if (!check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_)), "Creating the DXGI factory", error)) return false;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)))) {
        ComPtr<IDXGIAdapter> warp;
        if (!check(factory_->EnumWarpAdapter(IID_PPV_ARGS(&warp)), "Finding a Direct3D 12 adapter", error)) return false;
        if (!check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)), "Creating the Direct3D 12 device", error))
            return false;
    }
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!check(device_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)), "Creating the command queue", error)) return false;

    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kFrames;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    ComPtr<IDXGISwapChain1> sc;
    if (!check(factory_->CreateSwapChainForHwnd(queue_.Get(), hwnd, &sd, nullptr, nullptr, &sc), "Creating the swap chain", error)) return false;
    factory_->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    if (!check(sc.As(&swap_), "Querying the swap chain", error)) return false;

    D3D12_DESCRIPTOR_HEAP_DESC rh{};
    rh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rh.NumDescriptors = kFrames;
    if (!check(device_->CreateDescriptorHeap(&rh, IID_PPV_ARGS(&rtv_heap_)), "Creating the target heap", error)) return false;
    rtv_size_ = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_DESCRIPTOR_HEAP_DESC sh{};
    sh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    sh.NumDescriptors = 4;  // the game picture, the AI picture, the sprite atlas and the sprite references
    sh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (!check(device_->CreateDescriptorHeap(&sh, IID_PPV_ARGS(&srv_heap_)), "Creating the texture heap", error)) return false;

    for (UINT i = 0; i < kFrames; i++)
        if (!check(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators_[i])), "Creating an allocator", error))
            return false;
    if (!check(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(), nullptr, IID_PPV_ARGS(&list_)),
               "Creating the command list", error))
        return false;
    list_->Close();
    if (!check(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "Creating the fence", error)) return false;
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 4;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &range;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rd{};
    rd.NumParameters = 2;
    rd.pParameters = params;
    rd.NumStaticSamplers = 1;
    rd.pStaticSamplers = &sampler;
    ComPtr<ID3DBlob> blob, errors;
    if (!check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Building the root signature", error)) return false;
    if (!check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)), "Creating the root signature", error))
        return false;

    ComPtr<ID3DBlob> vs, ps;
    // A failed compile reports the compiler's own message, so the cause is visible without a debugger.
    auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>* out, const char* what) {
        errors.Reset();
        if (check(D3DCompile(kShader, sizeof(kShader) - 1, "picture", nullptr, nullptr, entry, target, 0, 0, out->GetAddressOf(), &errors), what, error))
            return true;
        if (errors) *error += std::string(" ") + static_cast<const char*>(errors->GetBufferPointer());
        return false;
    };
    if (!compile("vs", "vs_5_0", &vs, "Compiling the vertex shader")) return false;
    if (!compile("ps", "ps_5_0", &ps, "Compiling the pixel shader")) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root_.Get();
    pd.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pd.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = UINT_MAX;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    pd.SampleDesc.Count = 1;
    if (!check(device_->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&pso_)), "Creating the pipeline", error)) return false;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = kWidth;
    td.Height = kHeight;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    if (!check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture_)),
               "Creating the picture texture", error))
        return false;
    device_->CreateShaderResourceView(texture_.Get(), nullptr, srv_heap_->GetCPUDescriptorHandleForHeapStart());
    {
        // Until the AI model is loaded, the second slot holds an empty view, which reads as black.
        D3D12_SHADER_RESOURCE_VIEW_DESC empty{};
        empty.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        empty.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        empty.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        empty.Texture2D.MipLevels = 1;
        auto slot = srv_heap_->GetCPUDescriptorHandleForHeapStart();
        slot.ptr += device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        device_->CreateShaderResourceView(nullptr, &empty, slot);
    }

    {
        // Per screen pixel, the atlas cell of its upscaled sprite picture. The atlas itself takes 256 MB of video
        // memory, so it is only created once there is an upscaled picture to show (see create_atlas); until then
        // its slot holds an empty view.
        const UINT increment = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_RESOURCE_DESC refs_desc = td;
        refs_desc.Format = DXGI_FORMAT_R32_UINT;
        if (!check(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &refs_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&refs_)),
                   "Creating the sprite reference texture", error))
            return false;
        auto slot = srv_heap_->GetCPUDescriptorHandleForHeapStart();
        slot.ptr += 2 * SIZE_T(increment);
        D3D12_SHADER_RESOURCE_VIEW_DESC empty{};
        empty.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        empty.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        empty.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        empty.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(nullptr, &empty, slot);
        slot.ptr += increment;
        device_->CreateShaderResourceView(refs_.Get(), nullptr, slot);
    }

    D3D12_HEAP_PROPERTIES upload_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = kWidth * 4 * kHeight * 2;  // the picture, then the sprite references
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    for (UINT i = 0; i < kFrames; i++) {
        if (!check(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&upload_[i])),
                   "Creating an upload buffer", error))
            return false;
        D3D12_RANGE none{0, 0};
        upload_[i]->Map(0, &none, reinterpret_cast<void**>(&upload_ptr_[i]));
    }
    create_targets();
    return true;
}

bool Presenter::enable_ai(const std::filesystem::path& weights, std::string* error) {
    UINT64 frequency = 0;
    queue_->GetTimestampFrequency(&frequency);
    if (!device_ || !ai_.init(device_.Get(), frequency, weights, error)) return false;
    auto slot = srv_heap_->GetCPUDescriptorHandleForHeapStart();
    slot.ptr += device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device_->CreateShaderResourceView(ai_.result(), nullptr, slot);
    return true;
}

namespace {

// Reads a PNG (or any picture format Windows knows) as RGBA with the Windows Imaging Component.
bool load_picture(const std::filesystem::path& file, UINT* w, UINT* h, std::vector<uint8_t>* rgba) {
    static const bool com = [] {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    }();
    if (!com) return false;
    ComPtr<IWICImagingFactory> factory;
    ComPtr<IWICBitmapDecoder> decoder;
    ComPtr<IWICBitmapFrameDecode> frame;
    ComPtr<IWICFormatConverter> converter;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->CreateDecoderFromFilename(file.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder)) ||
        FAILED(decoder->GetFrame(0, &frame)) || FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom)) ||
        FAILED(converter->GetSize(w, h)))
        return false;
    rgba->resize(size_t(*w) * size_t(*h) * 4);
    return SUCCEEDED(converter->CopyPixels(nullptr, *w * 4, UINT(rgba->size()), rgba->data()));
}

// Doubles an RGBA picture with bilinear filtering, the sample positions lying between the source pixels.
std::vector<uint8_t> double_size(const std::vector<uint8_t>& in, UINT w, UINT h) {
    std::vector<uint8_t> out(size_t(w) * 2 * size_t(h) * 2 * 4);
    for (UINT y = 0; y < h * 2; y++) {
        const float sy = std::clamp((float(y) + 0.5f) / 2 - 0.5f, 0.0f, float(h - 1));
        const UINT y0 = UINT(sy), y1 = std::min(y0 + 1, h - 1);
        const float fy = sy - float(y0);
        for (UINT x = 0; x < w * 2; x++) {
            const float sx = std::clamp((float(x) + 0.5f) / 2 - 0.5f, 0.0f, float(w - 1));
            const UINT x0 = UINT(sx), x1 = std::min(x0 + 1, w - 1);
            const float fx = sx - float(x0);
            for (int c = 0; c < 4; c++) {
                auto at = [&](UINT px, UINT py) { return float(in[(size_t(py) * w + px) * 4 + size_t(c)]); };
                const float v = (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
                out[(size_t(y) * w * 2 + x) * 4 + size_t(c)] = uint8_t(v + 0.5f);
            }
        }
    }
    return out;
}

}  // namespace

bool Presenter::create_atlas() {
    if (atlas_) return true;
    if (atlas_failed_) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC ad{};
    ad.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    ad.Width = ad.Height = kAtlasCells * HdSprites::kScale;
    ad.DepthOrArraySize = 1;
    ad.MipLevels = 1;
    ad.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    ad.SampleDesc.Count = 1;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &ad, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&atlas_)))) {
        // Without the video memory for it the game simply keeps its other filters; the sprites stay as they are.
        atlas_failed_ = true;
        atlas_.Reset();
        return false;
    }
    // Earlier frames may still be reading the empty view in this slot, so they finish before it changes.
    wait_idle();
    auto slot = srv_heap_->GetCPUDescriptorHandleForHeapStart();
    slot.ptr += 2 * SIZE_T(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    device_->CreateShaderResourceView(atlas_.Get(), nullptr, slot);
    return true;
}

void Presenter::upload_hd(UINT frame) {
    if (!hd_) return;
    auto pending = hd_->take_pending();
    if (pending.empty()) return;
    if (!create_atlas()) {
        for (const auto& p : pending) hd_->failed(p.slot);
        return;
    }
    if (atlas_readable_) {
        const auto b = transition(atlas_.Get(), kPictureReadable, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->ResourceBarrier(1, &b);
    }
    const UINT k = HdSprites::kScale;
    // A large batch is spread over several frames: each frame loads pictures until about 16 MB have been
    // uploaded. The rest stays pending and comes in the next frames.
    size_t budget = size_t(16) << 20;
    for (const auto& p : pending) {
        // Sprites are packed in rows ("shelves") with a gap of one original pixel between them. The place is
        // found before the picture is read, so a full atlas costs no decoding.
        const UINT cw = UINT(p.width) + 1, ch = UINT(p.rows) + 1;
        if (cw > kAtlasCells - 2 || ch > kAtlasCells - 2) {
            hd_->failed(p.slot);  // larger than even an empty atlas
            continue;
        }
        UINT x = shelf_x_, y = shelf_y_, shelf_height = shelf_height_;
        if (x + cw > kAtlasCells - 1) {
            x = 1;
            y += shelf_height;
            shelf_height = 0;
        }
        if (y + ch > kAtlasCells - 1) {
            // The atlas is full. Whether it is emptied is decided by AtlasClearing (see hd_sprites.h). After
            // emptying, the tracker asks again for each picture as soon as it shows on screen. Otherwise this
            // picture keeps its original pixels and is asked for again the next time it shows, which costs no
            // more than this check.
            const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (!clearing_.overflow(now, hd_->on_screen_cells())) {
                hd_->no_room(p.slot);
                continue;
            }
            shelf_x_ = shelf_y_ = 1;
            shelf_height_ = 0;
            hd_->atlas_cleared();
            break;
        }
        const size_t bytes = size_t(p.width) * k * size_t(p.rows) * k * 4;
        if (bytes > budget && budget != (size_t(16) << 20)) break;
        budget -= std::min(budget, bytes);
        UINT w = 0, h = 0;
        std::vector<uint8_t> rgba;
        if (!load_picture(p.file, &w, &h, &rgba)) {
            hd_->failed(p.slot);
            continue;
        }
        // Pictures four times as large are accepted too and doubled smoothly to the atlas resolution.
        if (w * 2 == UINT(p.width) * k && h * 2 == UINT(p.rows) * k) {
            rgba = double_size(rgba, w, h);
            w *= 2;
            h *= 2;
        }
        if (w != UINT(p.width) * k || h != UINT(p.rows) * k) {
            hd_->failed(p.slot);
            continue;
        }

        const UINT pitch = (w * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
        D3D12_HEAP_PROPERTIES uh{};
        uh.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = UINT64(pitch) * h;
        bd.Height = 1;
        bd.DepthOrArraySize = 1;
        bd.MipLevels = 1;
        bd.SampleDesc.Count = 1;
        bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> buffer;
        uint8_t* mapped = nullptr;
        D3D12_RANGE none{0, 0};
        if (FAILED(device_->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer))) ||
            FAILED(buffer->Map(0, &none, reinterpret_cast<void**>(&mapped)))) {
            hd_->failed(p.slot);
            continue;
        }
        for (UINT row = 0; row < h; row++) std::memcpy(mapped + size_t(row) * pitch, rgba.data() + size_t(row) * w * 4, size_t(w) * 4);
        buffer->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = atlas_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = buffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, pitch};
        list_->CopyTextureRegion(&dst, x * k, y * k, 0, &src, nullptr);
        // The place is taken only now that the copy is recorded, so a picture that failed leaves no gap.
        shelf_x_ = x + cw;
        shelf_y_ = y;
        shelf_height_ = std::max(shelf_height, ch);
        atlas_uploads_[frame].push_back(buffer);
        hd_->placed(p.slot, uint16_t(x), uint16_t(y));
    }
    const auto b = transition(atlas_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kPictureReadable);
    list_->ResourceBarrier(1, &b);
    atlas_readable_ = true;
}

void Presenter::create_targets() {
    DXGI_SWAP_CHAIN_DESC1 sd{};
    swap_->GetDesc1(&sd);
    width_ = sd.Width;
    height_ = sd.Height;
    auto handle = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < kFrames; i++) {
        swap_->GetBuffer(i, IID_PPV_ARGS(&targets_[i]));
        device_->CreateRenderTargetView(targets_[i].Get(), nullptr, handle);
        handle.ptr += rtv_size_;
    }
}

void Presenter::wait_idle() {
    const UINT64 v = next_fence_++;
    queue_->Signal(fence_.Get(), v);
    if (fence_->GetCompletedValue() < v) {
        fence_->SetEventOnCompletion(v, fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

void Presenter::resize() {
    if (!swap_) return;
    RECT r;
    GetClientRect(hwnd_, &r);
    if (r.right - r.left <= 0 || r.bottom - r.top <= 0) return;
    wait_idle();
    for (auto& t : targets_) t.Reset();
    swap_->ResizeBuffers(kFrames, 0, 0, DXGI_FORMAT_B8G8R8A8_UNORM, 0);
    create_targets();
}

RECT Presenter::picture_rect() const {
    // The original picture filled a 4:3 monitor, so 320x200 is shown with tall pixels.
    RECT client;
    GetClientRect(hwnd_, &client);
    const float w = float(client.right), h = float(client.bottom);
    float pw = w, ph = w * 3.0f / 4.0f;
    if (ph > h) { ph = h; pw = h * 4.0f / 3.0f; }
    RECT r;
    r.left = LONG((w - pw) / 2);
    r.top = LONG((h - ph) / 2);
    r.right = LONG(r.left + pw);
    r.bottom = LONG(r.top + ph);
    return r;
}

void Presenter::draw(const uint32_t* rgba, const uint32_t* hd_refs, int filter, const std::filesystem::path* capture) {
    if (!swap_ || width_ == 0) return;
    const UINT i = swap_->GetCurrentBackBufferIndex();
    if (fence_->GetCompletedValue() < fence_values_[i]) {
        fence_->SetEventOnCompletion(fence_values_[i], fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
    atlas_uploads_[i].clear();
    const size_t picture_bytes = size_t(kWidth) * 4 * kHeight;
    std::memcpy(upload_ptr_[i], rgba, picture_bytes);
    std::memcpy(upload_ptr_[i] + picture_bytes, hd_refs, picture_bytes);
    allocators_[i]->Reset();
    list_->Reset(allocators_[i].Get(), pso_.Get());

    if (texture_readable_) {
        const D3D12_RESOURCE_BARRIER b[2] = {transition(texture_.Get(), kPictureReadable, D3D12_RESOURCE_STATE_COPY_DEST),
                                             transition(refs_.Get(), kPictureReadable, D3D12_RESOURCE_STATE_COPY_DEST)};
        list_->ResourceBarrier(2, b);
    }
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = texture_.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = upload_[i].Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1, kWidth * 4};
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    dst.pResource = refs_.Get();
    src.PlacedFootprint.Offset = picture_bytes;
    src.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32_UINT;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER barriers[3] = {
        transition(texture_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kPictureReadable),
        transition(refs_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kPictureReadable),
        transition(targets_[i].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET),
    };
    list_->ResourceBarrier(3, barriers);
    texture_readable_ = true;
    upload_hd(i);

    // The network runs only when the picture has changed since its last run.
    const bool use_ai = filter == kAiFilter && ai_.ready();
    if (use_ai) {
        const size_t pixels = size_t(kWidth) * kHeight;
        if (ai_input_.size() != pixels || std::memcmp(ai_input_.data(), rgba, pixels * 4) != 0) {
            ai_.record(list_.Get(), texture_.Get());
            ai_input_.assign(rgba, rgba + pixels);
        }
    }
    list_->SetPipelineState(pso_.Get());

    auto rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += SIZE_T(i) * rtv_size_;
    const float black[4] = {0, 0, 0, 1};
    list_->ClearRenderTargetView(rtv, black, 0, nullptr);
    list_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const RECT pr = picture_rect();
    D3D12_VIEWPORT vp{float(pr.left), float(pr.top), float(pr.right - pr.left), float(pr.bottom - pr.top), 0, 1};
    D3D12_RECT scissor{0, 0, LONG(width_), LONG(height_)};
    list_->RSSetViewports(1, &vp);
    list_->RSSetScissorRects(1, &scissor);
    list_->SetGraphicsRootSignature(root_.Get());
    ID3D12DescriptorHeap* heaps[] = {srv_heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    struct {
        float scale[2];
        uint32_t mode, unused;
    } params{{vp.Width / kWidth, vp.Height / kHeight}, uint32_t(filter), 0};
    list_->SetGraphicsRoot32BitConstants(0, 4, &params, 0);
    list_->SetGraphicsRootDescriptorTable(1, srv_heap_->GetGPUDescriptorHandleForHeapStart());
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list_->DrawInstanced(3, 1, 0, 0);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    if (capture) {
        // A screenshot copies the finished frame, so it shows the picture exactly as filtered on screen.
        const D3D12_RESOURCE_DESC desc = targets_[i]->GetDesc();
        UINT64 total = 0;
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
        if (!readback_ || readback_size_ < total) {
            D3D12_HEAP_PROPERTIES rh{};
            rh.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width = total;
            rd.Height = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            readback_.Reset();
            readback_size_ = 0;
            if (SUCCEEDED(device_->CreateCommittedResource(&rh, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                           IID_PPV_ARGS(&readback_))))
                readback_size_ = total;
        }
        if (!readback_) capture = nullptr;
    }
    if (capture && use_ai) ai_.record_readback(list_.Get());
    if (capture) {
        const auto to_copy = transition(targets_[i].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list_->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = targets_[i].Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION into{};
        into.pResource = readback_.Get();
        into.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        into.PlacedFootprint = footprint;
        list_->CopyTextureRegion(&into, 0, 0, 0, &from, nullptr);
        const auto back = transition(targets_[i].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
        list_->ResourceBarrier(1, &back);
    } else {
        const auto back = transition(targets_[i].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        list_->ResourceBarrier(1, &back);
    }
    list_->Close();
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    swap_->Present(1, 0);
    fence_values_[i] = next_fence_++;
    queue_->Signal(fence_.Get(), fence_values_[i]);
    if (capture) {
        if (fence_->GetCompletedValue() < fence_values_[i]) {
            fence_->SetEventOnCompletion(fence_values_[i], fence_event_);
            WaitForSingleObject(fence_event_, INFINITE);
        }
        uint8_t* data = nullptr;
        D3D12_RANGE range{0, SIZE_T(readback_size_)};
        if (SUCCEEDED(readback_->Map(0, &range, reinterpret_cast<void**>(&data)))) {
            const LONG w = std::max<LONG>(1, pr.right - pr.left), h = std::max<LONG>(1, pr.bottom - pr.top);
            std::vector<uint32_t> pixels(size_t(w) * size_t(h));
            for (LONG y = 0; y < h; y++) {
                const uint8_t* row = data + footprint.Offset + size_t(y + pr.top) * footprint.Footprint.RowPitch;
                for (LONG x = 0; x < w; x++) {
                    const uint8_t* px = row + size_t(x + pr.left) * 4;  // stored as blue, green, red, alpha
                    pixels[size_t(y) * size_t(w) + size_t(x)] = uint32_t(px[2]) | (uint32_t(px[1]) << 8) | (uint32_t(px[0]) << 16) | 0xFF000000u;
                }
            }
            D3D12_RANGE none{0, 0};
            readback_->Unmap(0, &none);
            write_png(*capture, pixels.data(), unsigned(w), unsigned(h), 1);
        }
        if (use_ai) {
            // With the AI filter a screenshot also keeps the network's input and its full-size result, so the
            // GPU output can be checked against tools/ai/reference_upscale.py.
            auto stem = capture->parent_path() / capture->stem();
            write_png(std::filesystem::path(stem.wstring() + L"-in.png"), ai_input_.data(), kWidth, kHeight, 1);
            ai_.save_readback(std::filesystem::path(stem.wstring() + L"-ai.png"));
        }
    }
}

}  // namespace nax
