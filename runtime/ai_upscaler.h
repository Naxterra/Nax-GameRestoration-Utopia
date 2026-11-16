// Runs a Real-ESRGAN SRVGGNetCompact network on the GPU with Direct3D 12 compute shaders, turning the 320x200
// game picture into a 1280x800 picture. The weights come from the file tools/ai/convert_realesrgan.py writes.
#pragma once

#include <windows.h>

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nax {

class AiUpscaler {
public:
    static constexpr UINT kInWidth = 320, kInHeight = 200;
    static constexpr UINT kScale = 4;
    static constexpr UINT kOutWidth = kInWidth * kScale, kOutHeight = kInHeight * kScale;

    bool init(ID3D12Device* device, UINT64 timestamp_frequency, const std::filesystem::path& weights, std::string* error);
    bool ready() const { return ready_; }

    // Records the network into the command list. The picture must be readable by compute shaders; afterwards
    // the result is readable by pixel shaders.
    void record(ID3D12GraphicsCommandList* list, ID3D12Resource* picture);
    ID3D12Resource* result() const { return result_.Get(); }

    // Records a copy of the result for saving, and writes it as a PNG file once the GPU has finished.
    void record_readback(ID3D12GraphicsCommandList* list);
    bool save_readback(const std::filesystem::path& path);

    // GPU time of the last finished run in milliseconds, or a negative value before the first run.
    double last_gpu_ms();

private:
    struct Layer {
        uint32_t weights, biases, slopes, in_channels, out_channels;
        bool prelu;
    };

    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> conv_, shuffle_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
    Microsoft::WRL::ComPtr<ID3D12Resource> weights_, weights_upload_, features_[2], result_, readback_;
    std::vector<Layer> layers_;
    UINT64 weights_bytes_ = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT readback_footprint_{};
    bool weights_copied_ = false;
    bool result_readable_ = false;  // the result is in the pixel shader state rather than the compute state
    bool ready_ = false;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> queries_;
    Microsoft::WRL::ComPtr<ID3D12Resource> query_results_;
    UINT64 timestamp_frequency_ = 0;
    bool timed_ = false;
};

}  // namespace nax
