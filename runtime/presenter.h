// Draws the 320x200 game picture into a window with Direct3D 12.
#pragma once

#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace nax {

class Presenter {
public:
    static constexpr UINT kWidth = 320;
    static constexpr UINT kHeight = 200;

    ~Presenter();
    bool init(HWND hwnd, std::string* error);
    void resize();
    static constexpr int kFilterCount = 3;  // sharp pixels, smooth edges, CRT
    // Draws one frame with the given filter; with a capture path it also saves the frame as a PNG file.
    void draw(const uint32_t* rgba, int filter, const std::filesystem::path* capture = nullptr);
    // The rectangle inside the client area that shows the game picture, for mapping mouse positions.
    RECT picture_rect() const;

private:
    static constexpr UINT kFrames = 2;
    void wait_idle();
    void create_targets();

    HWND hwnd_ = nullptr;
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> swap_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> rtv_heap_, srv_heap_;
    UINT rtv_size_ = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> targets_[kFrames];
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocators_[kFrames];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list_;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fence_values_[kFrames]{};
    UINT64 next_fence_ = 1;
    HANDLE fence_event_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pso_;
    Microsoft::WRL::ComPtr<ID3D12Resource> texture_;
    Microsoft::WRL::ComPtr<ID3D12Resource> upload_[kFrames];
    uint8_t* upload_ptr_[kFrames]{};
    bool texture_readable_ = false;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback_;
    UINT64 readback_size_ = 0;
    UINT width_ = 0, height_ = 0;
};

}  // namespace nax
