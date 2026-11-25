#include "presenter.h"

#include <d3dcompiler.h>
#include <wincodec.h>

#include <algorithm>
#include <cmath>
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
//   3  AI upscaling: where a sprite or screen has a high-resolution redrawn picture (see hd_sprites.h), that
//      picture is drawn at the full screen resolution with smooth alpha compositing. Bitmap fonts, cursors,
//      popups, and un-remastered sprites use crisp sharp-bilinear scaling so text stays razor-sharp and nothing
//      looks blurred or smeared.
const char kShader[] = R"(
cbuffer Params : register(b0) { float2 scale; uint mode; uint unused; float3 tint; float tint_pad; };
Texture2D picture : register(t0);
Texture2D upscaled : register(t1);
Texture2D atlas : register(t2);
Texture2D<uint> refs : register(t3);
Texture2D fallback_atlas : register(t4);
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

// For bitmap fonts, cursors, popups, and sprites that do not have an 8x redrawn picture yet, crisp
// sharp-bilinear keeps every 1-pixel letter and detail razor-sharp with zero blurry upscaler smearing.
float3 ai_shapes(float2 uv) { return sharp(uv); }

// A sprite's upscaled picture has a smooth anti-aliased outline, while the game's pixels have stair steps.
// At every screen point we inspect the sprite at the pixel underneath and its eight neighbours, tracking both
// the topmost layer and the highest layer beneath it, and blend the topmost sprite's smooth alpha over the
// underlying layer (or the AI-filtered fallback where nothing lies underneath).
float4 hd_part(uint r, float2 local) {
    float2 cell = float2(r & 0x3FF, (r >> 10) & 0x3FF);
    return atlas.SampleLevel(smooth, (cell + local) / 1024.0, 0);  // the atlas spans 1024 original pixels
}
int order_of(uint r) { return int((r >> 20) & 0x7FF); }
bool is_dimmed(uint r) { return (r >> 31) != 0; }
// True when drawing order a lies above b, with the count starting again after 2046 (0 is always below).
bool above(int a, int b) {
    if (b == 0) return a != 0;
    if (a == 0) return false;
    int d = (a - b + 2046) % 2046;
    return d != 0 && d < 1023;
}
float3 apply_night_tint(float3 col, float2 uv) {
    if (!any(tint < 0.995)) return col;
    float3 dos_col = sharp(uv);
    float dos_bright = max(dos_col.r, max(dos_col.g, dos_col.b));
    float hd_bright = max(col.r, max(col.g, col.b));
    float emissive = saturate((dos_bright - 0.72) * 3.5) * saturate((hd_bright - 0.50) * 3.0);
    return col * lerp(tint, float3(1, 1, 1), emissive);
}
float3 ai_picture(float2 uv) {
    float2 s = uv * size;
    int2 p = clamp(int2(floor(s)), int2(0, 0), int2(319, 199));
    float2 f = s - p;
    uint r = refs.Load(int3(p, 0));
    // Non-HD pixels (mouse cursor, HUD text, and masked popups) must stay on top of adjacent HD sprites/backdrops.
    if (r == 0xFFFFFFFFu || r == 0xFFFFFFFEu) return ai_shapes(uv);
    float4 own = hd_part(r, f);
    if (order_of(r) == 0 && own.a > 0.02) own.a = 1.0;
    int top_order = -1, under_order = -1;
    bool top_dim = false, under_dim = false;
    float4 top_px = float4(0, 0, 0, 0), under_px = float4(0, 0, 0, 0);
    if (own.a > 0.02) {
        top_order = order_of(r);
        top_dim = is_dimmed(r);
        top_px = own;
    }
    [unroll] for (int dy = -1; dy <= 1; dy++) {
        [unroll] for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            float2 local = f - float2(dx, dy);
            // Stay strictly inside the 1-cell transparent border so bilinear filtering never touches another sprite.
            if (any(local < -0.85) || any(local > 1.85)) continue;
            int2 q = p + int2(dx, dy);
            if (any(q < int2(0, 0)) || any(q > int2(319, 199))) continue;
            uint n = refs.Load(int3(q, 0));
            if (n == 0xFFFFFFFFu || n == 0xFFFFFFFEu) continue;
            float4 other = hd_part(n, local);
            if (other.a <= 0.02) continue;
            int o = order_of(n);
            bool od = is_dimmed(n);
            if (o == 0) other.a = 1.0;
            if (top_order < 0) {
                top_order = o;
                top_dim = od;
                top_px = other;
            } else if (o == top_order) {
                if (other.a > top_px.a) { top_px = other; top_dim = od; }
            } else if (above(o, top_order)) {
                under_order = top_order;
                under_dim = top_dim;
                under_px = top_px;
                top_order = o;
                top_dim = od;
                top_px = other;
            } else if (under_order < 0) {
                under_order = o;
                under_dim = od;
                under_px = other;
            } else if (o == under_order) {
                if (other.a > under_px.a) { under_px = other; under_dim = od; }
            } else if (above(o, under_order)) {
                under_order = o;
                under_dim = od;
                under_px = other;
            }
        }
    }
    if (top_order < 0) return ai_shapes(uv);
    float3 top_rgb = top_dim ? apply_night_tint(top_px.rgb, uv) : top_px.rgb;
    if (top_px.a >= 0.99) return top_rgb;
    float3 base_col;
    if (under_order >= 0 && under_px.a >= 0.99) {
        base_col = under_dim ? apply_night_tint(under_px.rgb, uv) : under_px.rgb;
    } else if (under_order >= 0 && under_px.a > 0.02) {
        float3 under_rgb = under_dim ? apply_night_tint(under_px.rgb, uv) : under_px.rgb;
        base_col = lerp(ai_shapes(uv), under_rgb, saturate(under_px.a));
    } else {
        base_col = ai_shapes(uv);
    }
    return lerp(base_col, top_rgb, saturate(top_px.a));
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

struct WVIn {
    float2 pos : POSITION;
    float2 uv : TEXCOORD0;
    float4 col : TEXCOORD1;
};
struct WVOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : TEXCOORD1;
};
WVOut world_vs(WVIn i) {
    WVOut o;
    o.pos = float4(i.pos, 0, 1);
    o.uv = i.uv;
    o.col = i.col;
    return o;
}
float3 popup_panel_bg(float2 uv) {
    // Sleek dark brushed-gunmetal UI panel gradient replacing Utopia's coarse 1992 grey 8x8 dither speckle.
    float grad = saturate(uv.y * 1.15);
    return lerp(float3(0.105, 0.125, 0.165), float3(0.068, 0.082, 0.112), grad);
}
bool is_grey_speckle(float3 c) {
    float hi = max(c.r, max(c.g, c.b));
    float lo = min(c.r, min(c.g, c.b));
    float lum = (c.r + c.g + c.b) / 3.0;
    return (hi - lo) < 0.045 && lum >= 0.19 && lum <= 0.435;
}
bool is_grey_speckle_tinted(float3 c) {
    if (!any(tint < 0.995)) return is_grey_speckle(c);
    float3 norm = c / max(tint, float3(0.35, 0.35, 0.35));
    float hi = max(norm.r, max(norm.g, norm.b));
    float lo = min(norm.r, min(norm.g, norm.b));
    float lum = (norm.r + norm.g + norm.b) / 3.0;
    return (hi - lo) < 0.055 && lum >= 0.15 && lum <= 0.435;
}
float4 world_ps(WVOut i) : SV_Target {
    // col.a encodes the quad type:
    //   >= 6.5          : L-shaped floating popup (like mode 6, plus discards right sidebar button zone 268..315, 79..158)
    //   >= 5.5 && < 6.5 : floating popup dialog (discards kPopupHole & order-0 backdrop, upgrades grey speckle to sleek dark slate)
    //   >= 4.5 && < 5.5 : docked HUD text strip with full-screen backdrop (order 0) keyed out so counters float cleanly
    //   >= 3.5 && < 4.5 : docked HUD strip using full ai_picture(uv)
    //   >= 2.5 && < 3.5 : un-remastered 32xN fallback sprite from fallback_atlas (crisp nearest texel)
    //   >= 1.5 && < 2.5 : 8x HD sprite from atlas with smooth bilinear alpha and optional night dimming
    //   < 1.5           : solid translucent UI backing bar or isometric hover highlight
    if (i.col.a >= 5.5) {
        float2 s = i.uv * size;
        int2 p = clamp(int2(floor(s)), int2(0, 0), int2(319, 199));
        if (i.col.a >= 6.5 && p.x >= 268 && p.x < 316 && p.y >= 79 && p.y < 159) discard;
        uint r = refs.Load(int3(p, 0));
        if (r == 0xFFFFFFFEu) discard;
        if (r != 0xFFFFFFFFu && order_of(r) == 0) discard;
        float3 panel = popup_panel_bg(i.uv);
        float2 f = s - p;
        if (r == 0xFFFFFFFFu) {
            float3 raw = texel(p);
            if (!is_grey_speckle_tinted(raw)) return float4(raw, 1.0);
        }
        // Two-layer HD compositing matching ai_picture(uv) over the sleek dark panel so multi-tile HD buildings
        // inside SELECT A BUILDING.. blend cleanly across internal tile seams and respect night dimming.
        int top_order = -1, under_order = -1;
        bool top_dim = false, under_dim = false;
        float4 top_px = float4(0, 0, 0, 0), under_px = float4(0, 0, 0, 0);
        if (r != 0xFFFFFFFFu) {
            float4 own = hd_part(r, f);
            if (own.a > 0.02) {
                top_order = order_of(r);
                top_dim = is_dimmed(r);
                top_px = own;
            }
        }
        [unroll] for (int dy = -1; dy <= 1; dy++) {
            [unroll] for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                float2 local = f - float2(dx, dy);
                if (any(local < -0.85) || any(local > 1.85)) continue;
                int2 q = p + int2(dx, dy);
                if (any(q < int2(0, 0)) || any(q > int2(319, 199))) continue;
                uint n = refs.Load(int3(q, 0));
                if (n == 0xFFFFFFFFu || n == 0xFFFFFFFEu || order_of(n) == 0) continue;
                float4 other = hd_part(n, local);
                if (other.a <= 0.02) continue;
                int o = order_of(n);
                bool od = is_dimmed(n);
                if (top_order < 0) {
                    top_order = o; top_dim = od; top_px = other;
                } else if (o == top_order) {
                    if (other.a > top_px.a) { top_px = other; top_dim = od; }
                } else if (above(o, top_order)) {
                    under_order = top_order; under_dim = top_dim; under_px = top_px;
                    top_order = o; top_dim = od; top_px = other;
                } else if (under_order < 0) {
                    under_order = o; under_dim = od; under_px = other;
                } else if (o == under_order) {
                    if (other.a > under_px.a) { under_px = other; under_dim = od; }
                } else if (above(o, under_order)) {
                    under_order = o; under_dim = od; under_px = other;
                }
            }
        }
        if (top_order < 0) return float4(panel, 1.0);
        float3 top_rgb = top_dim ? apply_night_tint(top_px.rgb, i.uv) : top_px.rgb;
        if (top_px.a >= 0.99) return float4(top_rgb, 1.0);
        float3 base_col = panel;
        if (under_order >= 0 && under_px.a >= 0.99) {
            base_col = under_dim ? apply_night_tint(under_px.rgb, i.uv) : under_px.rgb;
        } else if (under_order >= 0 && under_px.a > 0.02) {
            float3 under_rgb = under_dim ? apply_night_tint(under_px.rgb, i.uv) : under_px.rgb;
            base_col = lerp(panel, under_rgb, saturate(under_px.a));
        }
        return float4(lerp(base_col, top_rgb, saturate(top_px.a)), 1.0);
    }
    if (i.col.a >= 4.5) {
        int2 p = clamp(int2(floor(i.uv * size)), int2(0, 0), int2(319, 199));
        uint r = refs.Load(int3(p, 0));
        if (r != 0xFFFFFFFFu && order_of(r) == 0) discard;
        if (r == 0xFFFFFFFFu || r == 0xFFFFFFFEu) {
            float3 raw = texel(p);
            if (p.y >= 159 && (max(raw.r, max(raw.g, raw.b)) < 0.08 || is_grey_speckle_tinted(raw))) discard;
            return float4(raw, 1.0);
        }
        return float4(ai_picture(i.uv), 1.0);
    }
    if (i.col.a >= 3.5) {
        return float4(ai_picture(i.uv), 1.0);
    }
    if (i.col.a >= 2.5) {
        int2 p = clamp(int2(floor(i.uv * 1024.0)), int2(0, 0), int2(1023, 1023));
        float4 px = fallback_atlas.Load(int3(p, 0));
        if (px.a <= 0.02) discard;
        return px;
    }
    if (i.col.a >= 1.5) {
        float4 px = atlas.SampleLevel(smooth, i.uv, 0);
        if (px.a <= 0.02) discard;
        float3 rgb = px.rgb;
        if (any(i.col.rgb < 0.995)) {
            float bright = max(rgb.r, max(rgb.g, rgb.b));
            float warm = saturate((rgb.r + rgb.g * 0.85 - rgb.b * 1.1 - 0.35) * 3.0);
            float emissive = saturate((bright - 0.72) * 3.5) * warm;
            rgb *= lerp(i.col.rgb, float3(1, 1, 1), emissive);
        }
        return float4(rgb, px.a);
    }
    return i.col;
}
)";

struct WorldQuadVertex {
    float x, y;
    float u, v;
    float r, g, b, a;
};

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
    sh.NumDescriptors = 5;  // picture, AI picture, HD sprite atlas, sprite references, and world fallback atlas
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
    range.NumDescriptors = 5;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
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
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> blob, errors;
    if (!check(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors), "Building the root signature", error)) return false;
    if (!check(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)), "Creating the root signature", error))
        return false;

    ComPtr<ID3DBlob> vs, ps, wvs, wps;
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
    if (!compile("world_vs", "vs_5_0", &wvs, "Compiling the world vertex shader")) return false;
    if (!compile("world_ps", "ps_5_0", &wps, "Compiling the world pixel shader")) return false;
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

    D3D12_INPUT_ELEMENT_DESC world_layout[3]{};
    world_layout[0] = {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    world_layout[1] = {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    world_layout[2] = {"TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC wpd = pd;
    wpd.VS = {wvs->GetBufferPointer(), wvs->GetBufferSize()};
    wpd.PS = {wps->GetBufferPointer(), wps->GetBufferSize()};
    wpd.InputLayout = {world_layout, 3};
    auto& rt = wpd.BlendState.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
    rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    if (!check(device_->CreateGraphicsPipelineState(&wpd, IID_PPV_ARGS(&world_pso_)), "Creating the world pipeline", error)) return false;

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
        slot.ptr += increment;
        device_->CreateShaderResourceView(nullptr, &empty, slot);
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
    D3D12_RESOURCE_DESC wvbd = bd;
    wvbd.Width = UINT64(kMaxWorldVerts) * sizeof(WorldQuadVertex);
    for (UINT i = 0; i < kFrames; i++) {
        if (!check(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&upload_[i])),
                   "Creating an upload buffer", error))
            return false;
        D3D12_RANGE none{0, 0};
        upload_[i]->Map(0, &none, reinterpret_cast<void**>(&upload_ptr_[i]));
        if (!check(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &wvbd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&world_vb_[i])),
                   "Creating a world vertex buffer", error))
            return false;
        world_vb_[i]->Map(0, &none, reinterpret_cast<void**>(&world_vb_ptr_[i]));
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

bool Presenter::ensure_fallback_texture() {
    if (fallback_tex_) return true;
    if (fallback_failed_) return false;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC fd{};
    fd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    fd.Width = fd.Height = WorldFallbackAtlas::kSize;
    fd.DepthOrArraySize = 1;
    fd.MipLevels = 1;
    fd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    fd.SampleDesc.Count = 1;
    if (FAILED(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &fd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&fallback_tex_)))) {
        fallback_failed_ = true;
        fallback_tex_.Reset();
        return false;
    }
    D3D12_HEAP_PROPERTIES uh{};
    uh.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = UINT64(WorldFallbackAtlas::kSize) * WorldFallbackAtlas::kSize * 4;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_RANGE none{0, 0};
    for (UINT i = 0; i < kFrames; i++) {
        if (FAILED(device_->CreateCommittedResource(&uh, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&fallback_upload_[i]))) ||
            FAILED(fallback_upload_[i]->Map(0, &none, reinterpret_cast<void**>(&fallback_upload_ptr_[i])))) {
            fallback_failed_ = true;
            fallback_tex_.Reset();
            for (UINT j = 0; j < kFrames; j++) {
                fallback_upload_[j].Reset();
                fallback_upload_ptr_[j] = nullptr;
            }
            return false;
        }
    }
    wait_idle();
    auto slot = srv_heap_->GetCPUDescriptorHandleForHeapStart();
    slot.ptr += 4 * SIZE_T(device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV));
    device_->CreateShaderResourceView(fallback_tex_.Get(), nullptr, slot);
    return true;
}

bool Presenter::upload_hd(UINT frame) {
    if (!hd_) return false;
    auto pending = hd_->take_pending();
    if (pending.empty()) return false;
    if (!create_atlas()) {
        for (const auto& p : pending) hd_->failed(p.slot);
        return false;
    }
    if (atlas_readable_) {
        const auto b = transition(atlas_.Get(), kPictureReadable, D3D12_RESOURCE_STATE_COPY_DEST);
        list_->ResourceBarrier(1, &b);
    }
    const UINT k = HdSprites::kScale;
    // Allow a full screen backdrop plus all visible colony buildings to upload together in a single frame
    // so switching screens never flashes vanilla pixels before the HD artwork arrives.
    constexpr size_t kUploadBudget = size_t(64) << 20;
    size_t budget = kUploadBudget;
    size_t decode_budget = size_t(256) << 20;
    bool placed_any = false;
    for (const auto& p : pending) {
        // Sprites are packed in rows ("shelves") with a dedicated 1-cell border on all four sides (2 extra cells
        // in width and height) so adjacent sprites never overwrite each other's clamped-RGB transparent borders.
        const UINT cw = UINT(p.width) + 2, ch = UINT(p.rows) + 2;
        if (cw > kAtlasCells || ch > kAtlasCells) {
            hd_->failed(p.slot);  // larger than even an empty atlas
            continue;
        }
        UINT x = shelf_x_, y = shelf_y_, shelf_height = shelf_height_;
        if (x - 1 + cw > kAtlasCells) {
            x = 1;
            y += shelf_height;
            shelf_height = 0;
        }
        if (y - 1 + ch > kAtlasCells) {
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
        if (bytes > budget && budget != kUploadBudget) break;
        if (decode_budget == 0) break;
        UINT w = 0, h = 0;
        std::vector<uint8_t> rgba;
        const bool loaded = load_picture(p.file, &w, &h, &rgba);
        decode_budget -= std::min(decode_budget, std::max(rgba.size(), bytes / 4));
        if (!loaded) {
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
        budget -= std::min(budget, bytes);  // only pictures that will be uploaded count

        // Pad each uploaded sprite with a transparent 1-cell border on all four sides (sitting inside the 1-cell
        // atlas gaps around x, y) so neighbouring samples never read stale data after an atlas clear.
        const UINT upload_w = w + 2 * k, upload_h = h + 2 * k;
        const UINT pitch = (upload_w * 4 + D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1) & ~(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT - 1);
        D3D12_HEAP_PROPERTIES uh{};
        uh.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{};
        bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width = UINT64(pitch) * upload_h;
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
        // Pad each uploaded sprite with a 1-cell border on all four sides (clamped edge RGB, alpha = 0) so
        // bilinear filtering at sprite boundaries never darkens RGB toward black or reads stale atlas data.
        for (UINT by = 0; by < upload_h; by++) {
            const UINT sy = by < k ? 0 : (by >= k + h ? h - 1 : by - k);
            const uint8_t* src_row = rgba.data() + size_t(sy) * w * 4;
            uint8_t* dst_row = mapped + size_t(by) * pitch;
            const bool inside_y = by >= k && by < k + h;
            for (UINT bx = 0; bx < upload_w; bx++) {
                const UINT sx = bx < k ? 0 : (bx >= k + w ? w - 1 : bx - k);
                const uint8_t* sp = src_row + size_t(sx) * 4;
                uint8_t* dp = dst_row + size_t(bx) * 4;
                dp[0] = sp[0];
                dp[1] = sp[1];
                dp[2] = sp[2];
                dp[3] = (inside_y && bx >= k && bx < k + w) ? sp[3] : 0;
            }
        }
        buffer->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = atlas_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = buffer.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, upload_w, upload_h, 1, pitch};
        list_->CopyTextureRegion(&dst, (x - 1) * k, (y - 1) * k, 0, &src, nullptr);
        // The place is taken only now that the copy is recorded, so a picture that failed leaves no gap.
        shelf_x_ = x + cw;
        shelf_y_ = y;
        shelf_height_ = std::max(shelf_height, ch);
        atlas_uploads_[frame].push_back(buffer);
        hd_->placed(p.slot, uint16_t(x), uint16_t(y));
        placed_any = true;
    }
    const auto b = transition(atlas_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kPictureReadable);
    list_->ResourceBarrier(1, &b);
    atlas_readable_ = true;
    return placed_any;
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

void Presenter::draw(const uint32_t* rgba, const uint32_t* hd_refs, int filter, const std::filesystem::path* capture,
                     const std::function<void()>& refresh_refs, const WorldRenderData* world) {
    if (!swap_ || width_ == 0) return;
    const UINT i = swap_->GetCurrentBackBufferIndex();
    if (fence_->GetCompletedValue() < fence_values_[i]) {
        fence_->SetEventOnCompletion(fence_values_[i], fence_event_);
        WaitForSingleObject(fence_event_, INFINITE);
    }
    atlas_uploads_[i].clear();
    allocators_[i]->Reset();
    list_->Reset(allocators_[i].Get(), pso_.Get());
    // Upload pending HD pictures before copying hd_refs so freshly loaded sprites and screens appear on
    // this exact frame instead of flashing vanilla pixels for one or more frames first.
    if (upload_hd(i) && refresh_refs) {
        refresh_refs();
        if (upload_hd(i)) refresh_refs();
    }
    const size_t picture_bytes = size_t(kWidth) * 4 * kHeight;
    std::memcpy(upload_ptr_[i], rgba, picture_bytes);
    std::memcpy(upload_ptr_[i] + picture_bytes, hd_refs, picture_bytes);

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

    const bool world_pass = world != nullptr && !world->tiles.empty() && filter == kAiFilter && world_pso_ && world_vb_ptr_[i];
    if (world_pass && world->fallback && world->fallback->version() > 1 &&
        world->fallback->version() != uploaded_fallback_version_ && ensure_fallback_texture()) {
        if (fallback_readable_) {
            const auto b = transition(fallback_tex_.Get(), kPictureReadable, D3D12_RESOURCE_STATE_COPY_DEST);
            list_->ResourceBarrier(1, &b);
        }
        const size_t fb_bytes = size_t(WorldFallbackAtlas::kSize) * WorldFallbackAtlas::kSize * 4;
        std::memcpy(fallback_upload_ptr_[i], world->fallback->pixels(), fb_bytes);
        D3D12_TEXTURE_COPY_LOCATION fdst{};
        fdst.pResource = fallback_tex_.Get();
        fdst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION fsrc{};
        fsrc.pResource = fallback_upload_[i].Get();
        fsrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        fsrc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, WorldFallbackAtlas::kSize, WorldFallbackAtlas::kSize, 1,
                                          WorldFallbackAtlas::kSize * 4};
        list_->CopyTextureRegion(&fdst, 0, 0, 0, &fsrc, nullptr);
        const auto b = transition(fallback_tex_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kPictureReadable);
        list_->ResourceBarrier(1, &b);
        fallback_readable_ = true;
        uploaded_fallback_version_ = world->fallback->version();
    }

    // The live pixel shader uses crisp sharp-bilinear for un-remastered pixels and bitmap fonts, so the
    // neural network only needs to run when saving a diagnostic screenshot with -ai.png output.
    const bool use_ai = filter == kAiFilter && ai_.ready() && capture != nullptr && !world_pass;
    if (use_ai) {
        const size_t pixels = size_t(kWidth) * kHeight;
        AiUpscaler::Rect changed = AiUpscaler::kWhole;
        if (ai_input_.size() == pixels) {
            changed = {int(kWidth), int(kHeight), 0, 0};
            for (int y = 0; y < int(kHeight); y++) {
                const uint32_t* now = rgba + size_t(y) * kWidth;
                const uint32_t* before = ai_input_.data() + size_t(y) * kWidth;
                if (std::memcmp(now, before, kWidth * 4) == 0) continue;
                for (int x = 0; x < int(kWidth); x++)
                    if (now[x] != before[x]) {
                        changed = {std::min(changed.x0, x), std::min(changed.y0, y), std::max(changed.x1, x + 1), y + 1};
                    }
            }
        }
        if (changed.x0 < changed.x1) {
            ai_.record(list_.Get(), texture_.Get(), changed);
            ai_input_.assign(rgba, rgba + pixels);
        }
    }

    auto rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += SIZE_T(i) * rtv_size_;
    const float bg_colour[4] = {world_pass ? 0.025f : 0.0f, world_pass ? 0.030f : 0.0f, world_pass ? 0.045f : 0.0f, 1.0f};
    list_->ClearRenderTargetView(rtv, bg_colour, 0, nullptr);
    list_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const RECT pr = world_pass ? RECT{0, 0, LONG(width_), LONG(height_)} : picture_rect();
    D3D12_RECT scissor{0, 0, LONG(width_), LONG(height_)};
    list_->RSSetScissorRects(1, &scissor);
    list_->SetGraphicsRootSignature(root_.Get());
    ID3D12DescriptorHeap* heaps[] = {srv_heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    float tint[3] = {1.0f, 1.0f, 1.0f};
    if (hd_ && filter == kAiFilter) hd_->ambient_tint(tint);
    list_->SetGraphicsRootDescriptorTable(1, srv_heap_->GetGPUDescriptorHandleForHeapStart());
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    if (world_pass) {
        D3D12_VIEWPORT vp{0.0f, 0.0f, float(width_), float(height_), 0, 1};
        list_->RSSetViewports(1, &vp);
        float hud_sx = world->tile_scale_x, hud_sy = world->tile_scale_y;
        if (!world->hud_strips.empty() && world->hud_strips[0].src_w > 0 && world->hud_strips[0].src_h > 0) {
            hud_sx = float(world->hud_strips[0].dst.right - world->hud_strips[0].dst.left) / float(world->hud_strips[0].src_w);
            hud_sy = float(world->hud_strips[0].dst.bottom - world->hud_strips[0].dst.top) / float(world->hud_strips[0].src_h);
        }
        struct {
            float scale[2];
            uint32_t mode, unused;
            float tint[3], tint_pad;
        } params{{hud_sx, hud_sy}, uint32_t(filter), 0, {tint[0], tint[1], tint[2]}, 0.0f};
        list_->SetGraphicsRoot32BitConstants(0, 8, &params, 0);

        auto* verts = reinterpret_cast<WorldQuadVertex*>(world_vb_ptr_[i]);
        UINT vert_count = 0;
        const float inv_w = 2.0f / float(width_);
        const float inv_h = 2.0f / float(height_);
        auto ndc_x = [inv_w](float px) { return px * inv_w - 1.0f; };
        auto ndc_y = [inv_h](float py) { return 1.0f - py * inv_h; };
        auto push_quad = [&](float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1,
                             float r, float g, float b, float a) {
            if (vert_count + 6 > kMaxWorldVerts) return;
            const float lx = ndc_x(x0), rx = ndc_x(x1), ty = ndc_y(y0), by = ndc_y(y1);
            verts[vert_count++] = {lx, ty, u0, v0, r, g, b, a};
            verts[vert_count++] = {rx, ty, u1, v0, r, g, b, a};
            verts[vert_count++] = {rx, by, u1, v1, r, g, b, a};
            verts[vert_count++] = {lx, ty, u0, v0, r, g, b, a};
            verts[vert_count++] = {rx, by, u1, v1, r, g, b, a};
            verts[vert_count++] = {lx, by, u0, v1, r, g, b, a};
        };
        auto push_tri = [&](float x0, float y0, float x1, float y1, float x2, float y2, float r, float g, float b, float a) {
            if (vert_count + 3 > kMaxWorldVerts) return;
            verts[vert_count++] = {ndc_x(x0), ndc_y(y0), 0, 0, r, g, b, a};
            verts[vert_count++] = {ndc_x(x1), ndc_y(y1), 0, 0, r, g, b, a};
            verts[vert_count++] = {ndc_x(x2), ndc_y(y2), 0, 0, r, g, b, a};
        };

        // 1. Isometric planet map tiles and dynamic sprites in back-to-front painter's order.
        // Reserve 320 vertices so hover highlight, floating popup, docked HUD strips, and live cursor always render.
        constexpr UINT kReservedHudVerts = 320;
        for (const auto& wt : world->tiles) {
            if (vert_count + 6 > kMaxWorldVerts - kReservedHudVerts) break;
            float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
            WorldView::project_tile(*world, wt.row, wt.col, wt.width, wt.rows, wt.pixel_dx, wt.pixel_dy, &x0, &y0, &x1, &y1);
            // Expand flat 32x15 ground diamonds slightly (~0.45 DOS px) so bilinear edge alpha never leaves faint
            // grid lines or horizontal corner dashes between adjacent tiles over the dark clear colour.
            if (wt.width == 32 && wt.rows == 15) {
                const float pad_x = 0.40f * world->tile_scale_x;
                const float pad_y = 0.50f * world->tile_scale_y;
                x0 -= pad_x;
                x1 += pad_x;
                y0 -= pad_y;
                y1 += pad_y;
            }
            if (x1 < 0.0f || x0 > float(width_) || y1 < 0.0f || y0 > float(height_)) continue;
            if (wt.has_hd && atlas_) {
                const float u0 = float(wt.atlas_x) / float(kAtlasCells);
                const float v0 = float(wt.atlas_y) / float(kAtlasCells);
                const float u1 = float(wt.atlas_x + wt.width) / float(kAtlasCells);
                const float v1 = float(wt.atlas_y + wt.rows) / float(kAtlasCells);
                const float tr = wt.dimmed ? world->ambient_tint[0] : 1.0f;
                const float tg = wt.dimmed ? world->ambient_tint[1] : 1.0f;
                const float tb = wt.dimmed ? world->ambient_tint[2] : 1.0f;
                push_quad(x0, y0, x1, y1, u0, v0, u1, v1, tr, tg, tb, 2.0f);
            } else if (wt.fallback_id != 0 && world->fallback && fallback_tex_) {
                if (const auto* e = world->fallback->entry(wt.fallback_id)) {
                    const float inv_fb = 1.0f / float(WorldFallbackAtlas::kSize);
                    push_quad(x0, y0, x1, y1, float(e->x) * inv_fb, float(e->y) * inv_fb,
                              float(e->x + e->w) * inv_fb, float(e->y + e->h) * inv_fb, 1.0f, 1.0f, 1.0f, 3.0f);
                }
            }
        }

        // 2. Subtle translucent isometric diamond highlight under the hovered map tile (hidden while a popup is open).
        if (world->has_hover && !world->has_popup) {
            const float dr = float(world->hover_row) - world->center_row;
            const float dc = float(world->hover_col) - world->center_col;
            const float cx = world->origin_x + (dc - dr) * 16.0f * world->tile_scale_x;
            const float cy = world->origin_y + (dc + dr) * 8.0f * world->tile_scale_y;
            const float hw = 16.0f * world->tile_scale_x;
            const float hh = 7.5f * world->tile_scale_y;
            const float tx = cx, ty = cy - hh;
            const float rx = cx + hw, ry = cy;
            const float bx = cx, by = cy + hh;
            const float lx = cx - hw, ly = cy;
            push_tri(tx, ty, rx, ry, bx, by, 0.55f, 0.85f, 1.0f, 0.16f);
            push_tri(tx, ty, bx, by, lx, ly, 0.55f, 0.85f, 1.0f, 0.16f);
            auto push_edge = [&](float ax, float ay, float px, float py) {
                const float ex = px - ax, ey = py - ay;
                const float len = std::max(1.0f, std::sqrt(ex * ex + ey * ey));
                const float nx = (-ey / len) * 1.1f, ny = (ex / len) * 1.1f;
                push_tri(ax - nx, ay - ny, px - nx, py - ny, px + nx, py + ny, 0.78f, 0.95f, 1.0f, 0.65f);
                push_tri(ax - nx, ay - ny, px + nx, py + ny, ax + nx, ay + ny, 0.78f, 0.95f, 1.0f, 0.65f);
            };
            push_edge(tx, ty, rx, ry);
            push_edge(rx, ry, bx, by);
            push_edge(bx, by, lx, ly);
            push_edge(lx, ly, tx, ty);
        }

        // 3. When an in-game popup (SELECT A BUILDING.., MISSILE LAUNCHER, Trade, Intelligence Reports, etc.) is open,
        // dim the live ultrawide colony behind it and float ONLY the popup cleanly with sleek modern dark panels.
        if (world->has_popup && world->popup_src_w > 0 && world->popup_src_h > 0) {
            push_quad(0.0f, 0.0f, float(width_), float(height_), 0, 0, 0, 0, 0.015f, 0.020f, 0.035f, 0.54f);
            const float px0 = float(world->popup_dst.left), py0 = float(world->popup_dst.top);
            const float px1 = float(world->popup_dst.right), py1 = float(world->popup_dst.bottom);
            if (!world->popup_has_holes) {
                // Draw drop shadow and 2px metallic frame strictly around the outer perimeter so any discarded
                // order-0 pixel inside the popup rect never exposes bright border steel behind it.
                const float bw = 2.0f;
                const float sx0 = px0 - 6.0f, sy0 = py0 - 6.0f, sx1 = px1 + 8.0f, sy1 = py1 + 8.0f;
                push_quad(sx0, sy0, sx1, py0 - bw, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.60f);
                push_quad(sx0, py1 + bw, sx1, sy1, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.60f);
                push_quad(sx0, py0 - bw, px0 - bw, py1 + bw, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.60f);
                push_quad(px1 + bw, py0 - bw, sx1, py1 + bw, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.60f);
                push_quad(px0 - bw, py0 - bw, px1 + bw, py0, 0, 0, 0, 0, 0.34f, 0.39f, 0.48f, 0.95f);
                push_quad(px0 - bw, py1, px1 + bw, py1 + bw, 0, 0, 0, 0, 0.34f, 0.39f, 0.48f, 0.95f);
                push_quad(px0 - bw, py0, px0, py1, 0, 0, 0, 0, 0.34f, 0.39f, 0.48f, 0.95f);
                push_quad(px1, py0, px1 + bw, py1, 0, 0, 0, 0, 0.34f, 0.39f, 0.48f, 0.95f);
            }
            const float u0 = float(world->popup_src_x) / float(kWidth);
            const float v0 = float(world->popup_src_y) / float(kHeight);
            const float u1 = float(world->popup_src_x + world->popup_src_w) / float(kWidth);
            const float v1 = float(world->popup_src_y + world->popup_src_h) / float(kHeight);
            push_quad(px0, py0, px1, py1, u0, v0, u1, v1, 1.0f, 1.0f, 1.0f, world->popup_has_holes ? 7.0f : 6.0f);
        }

        // 4. Sleek compact dark frames for the docked HUD strips (no full-width black wings).
        const bool hide_top = world->has_popup && (world->popup_src_y < 14);
        const bool hide_bottom =
            (world->has_popup && (world->popup_src_y + world->popup_src_h > 170) && !world->has_warning_banner) ||
            !world->has_bottom_message;
        for (size_t si = 0; si < world->hud_strips.size(); si++) {
            if (world->has_popup && si == 4) continue;  // hide scroll-arrow pad while a modal popup is open
            if (hide_top && (si == 0 || si == 2)) continue;  // hide top bar & compass when an advisor report covers them
            if (hide_bottom && si == 1) continue;       // hide bottom bar when empty or overlapped by a tall popup
            HudStrip s = world->hud_strips[si];
            if (si == 1) WorldView::fit_bottom_strip_for_message(s, world);
            if (si == 3) WorldView::crop_sidebar_for_popup(s, world);
            if (s.solid_backdrop) {
                push_quad(0.0f, float(s.dst.top), float(width_), float(s.dst.bottom), 0, 0, 0, 0, 0.04f, 0.045f, 0.06f, 1.0f);
                const float line_y = s.dst.top == 0 ? float(s.dst.bottom) : float(s.dst.top) - 1.5f;
                push_quad(0.0f, line_y, float(width_), line_y + 1.5f, 0, 0, 0, 0, 0.30f, 0.35f, 0.44f, 0.90f);
            } else if (s.diamond_clip) {
                const float cx = 0.5f * float(s.dst.left + s.dst.right);
                const float cy = 0.5f * float(s.dst.top + s.dst.bottom);
                const float l = float(s.dst.left) - 4.0f, r = float(s.dst.right) + 4.0f;
                const float t = float(s.dst.top) - 3.0f, b = float(s.dst.bottom) + 3.0f;
                push_tri(cx, t, r, cy, cx, b, 0.03f, 0.035f, 0.05f, 0.94f);
                push_tri(cx, t, cx, b, l, cy, 0.03f, 0.035f, 0.05f, 0.94f);
            } else if (s.key_out_backdrop) {
                // Compact floating dark-glass status or warning pill with crisp metallic (or amber warning) border.
                const bool warn_pill = (si == 1 && world->has_warning_banner);
                const float x0 = float(s.dst.left) - 12.0f, x1 = float(s.dst.right) + 12.0f;
                const float y0 = float(s.dst.top) - 2.0f, y1 = float(s.dst.bottom) + 2.0f;
                push_quad(x0 - 1.5f, y0 - 1.5f, x1 + 1.5f, y1 + 1.5f, 0, 0, 0, 0,
                          warn_pill ? 0.82f : 0.30f, warn_pill ? 0.38f : 0.35f, warn_pill ? 0.22f : 0.44f,
                          warn_pill ? 0.94f : 0.85f);
                push_quad(x0, y0, x1, y1, 0, 0, 0, 0,
                          warn_pill ? 0.11f : 0.055f, warn_pill ? 0.055f : 0.065f, warn_pill ? 0.055f : 0.090f, 0.92f);
            } else {
                push_quad(float(s.dst.left) - 2.0f, float(s.dst.top) - 2.0f, float(s.dst.right) + 2.0f, float(s.dst.bottom) + 2.0f,
                          0, 0, 0, 0, 0.03f, 0.035f, 0.05f, 0.92f);
            }
        }

        // 5. Docked HD HUD strips sampled straight through ai_picture(uv) so icons and bitmap fonts stay crisp.
        for (size_t si = 0; si < world->hud_strips.size(); si++) {
            if (world->has_popup && si == 4) continue;
            if (hide_top && (si == 0 || si == 2)) continue;
            if (hide_bottom && si == 1) continue;
            HudStrip s = world->hud_strips[si];
            if (si == 1) WorldView::fit_bottom_strip_for_message(s, world);
            if (si == 3) WorldView::crop_sidebar_for_popup(s, world);
            const float u0 = float(s.src_x) / float(kWidth);
            const float v0 = float(s.src_y) / float(kHeight);
            const float u1 = float(s.src_x + s.src_w) / float(kWidth);
            const float v1 = float(s.src_y + s.src_h) / float(kHeight);
            const float mode_a = s.key_out_backdrop ? 5.0f : 4.0f;
            if (s.diamond_clip) {
                if (vert_count + 6 <= kMaxWorldVerts) {
                    const float lx = ndc_x(float(s.dst.left)), rx = ndc_x(float(s.dst.right));
                    const float ty = ndc_y(float(s.dst.top)), by = ndc_y(float(s.dst.bottom));
                    const float cx = 0.5f * (lx + rx), cy = 0.5f * (ty + by);
                    const float uc = 0.5f * (u0 + u1), vc = 0.5f * (v0 + v1);
                    verts[vert_count++] = {cx, ty, uc, v0, 1.0f, 1.0f, 1.0f, mode_a};
                    verts[vert_count++] = {rx, cy, u1, vc, 1.0f, 1.0f, 1.0f, mode_a};
                    verts[vert_count++] = {cx, by, uc, v1, 1.0f, 1.0f, 1.0f, mode_a};
                    verts[vert_count++] = {cx, ty, uc, v0, 1.0f, 1.0f, 1.0f, mode_a};
                    verts[vert_count++] = {cx, by, uc, v1, 1.0f, 1.0f, 1.0f, mode_a};
                    verts[vert_count++] = {lx, cy, u0, vc, 1.0f, 1.0f, 1.0f, mode_a};
                }
            } else {
                push_quad(float(s.dst.left), float(s.dst.top), float(s.dst.right), float(s.dst.bottom), u0, v0, u1, v1,
                          1.0f, 1.0f, 1.0f, mode_a);
            }
        }

        // 6. Utopia's live 10x10 game cursor (decoded from ds:0x5e68) rendered on top at native monitor refresh rate.
        if (world->has_game_cursor && world->cursor_fallback_id != 0 && world->fallback && fallback_tex_) {
            if (const auto* e = world->fallback->entry(world->cursor_fallback_id)) {
                const float inv_fb = 1.0f / float(WorldFallbackAtlas::kSize);
                const float cw = float(e->w) * world->cursor_scale_x;
                const float ch = float(e->h) * world->cursor_scale_y;
                push_quad(world->cursor_x, world->cursor_y, world->cursor_x + cw, world->cursor_y + ch,
                          float(e->x) * inv_fb, float(e->y) * inv_fb,
                          float(e->x + e->w) * inv_fb, float(e->y + e->h) * inv_fb, 1.0f, 1.0f, 1.0f, 3.0f);
            }
        }

        if (vert_count > 0) {
            D3D12_VERTEX_BUFFER_VIEW vbv{};
            vbv.BufferLocation = world_vb_[i]->GetGPUVirtualAddress();
            vbv.StrideInBytes = sizeof(WorldQuadVertex);
            vbv.SizeInBytes = vert_count * sizeof(WorldQuadVertex);
            list_->SetPipelineState(world_pso_.Get());
            list_->IASetVertexBuffers(0, 1, &vbv);
            list_->DrawInstanced(vert_count, 1, 0, 0);
        }
    } else {
        list_->SetPipelineState(pso_.Get());
        D3D12_VIEWPORT vp{float(pr.left), float(pr.top), float(pr.right - pr.left), float(pr.bottom - pr.top), 0, 1};
        list_->RSSetViewports(1, &vp);
        struct {
            float scale[2];
            uint32_t mode, unused;
            float tint[3], tint_pad;
        } params{{vp.Width / kWidth, vp.Height / kHeight}, uint32_t(filter), 0, {tint[0], tint[1], tint[2]}, 0.0f};
        list_->SetGraphicsRoot32BitConstants(0, 8, &params, 0);
        list_->DrawInstanced(3, 1, 0, 0);
    }
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
