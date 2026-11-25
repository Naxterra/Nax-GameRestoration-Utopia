#include "hd_sprites.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "machine.h"
#include "png.h"

namespace nax {

namespace {

constexpr uint32_t kVideo = 0xA0000;
constexpr int kPixels = 320 * 200;
constexpr int64_t kSettleMs = 150;
constexpr size_t kMaxDeferred = 32;
// The 64-bit FNV-1a hash that identifies each sprite.
constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

// Moves `from` to `to` when `to` does not exist yet. Returns whether it was moved.
bool moved(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::error_code ec;
    if (std::filesystem::exists(to, ec) || !std::filesystem::exists(from, ec)) return false;
    std::filesystem::rename(from, to, ec);
    return !ec;
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint8_t nibble(const Machine& m, uint32_t row, int x) {
    const uint8_t byte = m.mem[(row + uint32_t(x / 2)) & 0xFFFFF];
    return (x & 1) ? (byte & 0x0F) : (byte >> 4);
}

uint32_t colour_of(const Machine& m, uint8_t index) {
    auto expand = [](uint8_t v) { return uint32_t((v << 2) | (v >> 4)); };
    const uint8_t* d = m.dac[index];
    return expand(d[0]) | (expand(d[1]) << 8) | (expand(d[2]) << 16) | 0xFF000000u;
}

// The colours of the given palette entries, as one number.
uint64_t palette_of_dac(const std::array<std::array<uint8_t, 3>, 256>& dac, const std::bitset<256>& colours) {
    uint64_t h = kFnvBasis;
    for (int c = 0; c < 256; c++)
        if (colours[size_t(c)])
            for (int k = 0; k < 3; k++) h = (h ^ dac[size_t(c)][size_t(k)]) * kFnvPrime;
    return h;
}

uint64_t palette_of(const Machine& m, const std::bitset<256>& colours) {
    uint64_t h = kFnvBasis;
    for (int c = 0; c < 256; c++)
        if (colours[size_t(c)])
            for (int k = 0; k < 3; k++) h = (h ^ m.dac[c][k]) * kFnvPrime;
    return h;
}

std::array<std::array<uint8_t, 3>, 256> copy_dac(const Machine& m) {
    std::array<std::array<uint8_t, 3>, 256> out{};
    for (int c = 0; c < 256; c++)
        for (int k = 0; k < 3; k++) out[size_t(c)][size_t(k)] = m.dac[c][k];
    return out;
}

// Computes a palette-independent shape hash from an RGBA picture, together with the 6-bit RGB of each
// distinct non-transparent colour in first-appearance order.
uint64_t canonical_shape(const std::vector<uint32_t>& rgba, int width, int rows, std::vector<std::array<uint8_t, 3>>* palette_out) {
    palette_out->clear();
    std::vector<uint32_t> seen_rgb;
    uint64_t h = kFnvBasis;
    auto mix = [&](uint8_t v) { h = (h ^ v) * kFnvPrime; };
    mix(uint8_t(width));
    mix(uint8_t(width >> 8));
    mix(uint8_t(rows));
    mix(uint8_t(rows >> 8));
    for (uint32_t px : rgba) {
        if ((px >> 24) == 0) {
            mix(0);
            continue;
        }
        const uint32_t rgb = px & 0x00FFFFFFu;
        size_t idx = 0;
        while (idx < seen_rgb.size() && seen_rgb[idx] != rgb) idx++;
        if (idx == seen_rgb.size()) {
            if (seen_rgb.size() >= 64) return 0;
            seen_rgb.push_back(rgb);
            palette_out->push_back({uint8_t((rgb & 0xFF) >> 2), uint8_t(((rgb >> 8) & 0xFF) >> 2), uint8_t(((rgb >> 16) & 0xFF) >> 2)});
        }
        mix(uint8_t(idx + 1));
    }
    return h;
}

// Computes a palette-independent 256-colour shape hash for opaque full-screen backdrops (such as main_scr.vgg
// and option.vgg), returning the 6-bit RGB of each distinct colour in first-appearance order.
uint64_t canonical_image_shape_rgba(const std::vector<uint32_t>& rgba, int width, int rows,
                                    std::vector<std::array<uint8_t, 3>>* palette_out) {
    palette_out->clear();
    std::unordered_map<uint32_t, uint8_t> seen;
    seen.reserve(256);
    uint64_t h = kFnvBasis;
    auto mix = [&](uint8_t v) { h = (h ^ v) * kFnvPrime; };
    mix(0xFD);
    mix(uint8_t(width));
    mix(uint8_t(width >> 8));
    mix(uint8_t(rows));
    mix(uint8_t(rows >> 8));
    for (uint32_t px : rgba) {
        const uint8_t r6 = uint8_t((px & 0xFF) >> 2);
        const uint8_t g6 = uint8_t(((px >> 8) & 0xFF) >> 2);
        const uint8_t b6 = uint8_t(((px >> 16) & 0xFF) >> 2);
        const uint32_t key = uint32_t(r6) | (uint32_t(g6) << 6) | (uint32_t(b6) << 12);
        auto it = seen.find(key);
        uint8_t idx = 0;
        if (it == seen.end()) {
            if (palette_out->size() >= 256) return 0;
            idx = uint8_t(palette_out->size());
            seen.emplace(key, idx);
            palette_out->push_back({r6, g6, b6});
        } else {
            idx = it->second;
        }
        mix(idx);
    }
    return h;
}

uint64_t canonical_image_shape_indexed(const std::vector<uint8_t>& pixels, int width, int rows,
                                       const std::array<std::array<uint8_t, 3>, 256>& dac,
                                       std::vector<std::array<uint8_t, 3>>* palette_out,
                                       std::array<int16_t, 256>* dac_to_canon = nullptr) {
    palette_out->clear();
    std::array<int16_t, 256> map;
    map.fill(-1);
    std::array<uint32_t, 256> seen_keys{};
    uint64_t h = kFnvBasis;
    auto mix = [&](uint8_t v) { h = (h ^ v) * kFnvPrime; };
    mix(0xFD);
    mix(uint8_t(width));
    mix(uint8_t(width >> 8));
    mix(uint8_t(rows));
    mix(uint8_t(rows >> 8));
    for (uint8_t c : pixels) {
        int16_t idx = map[c];
        if (idx < 0) {
            const uint8_t r6 = dac[c][0], g6 = dac[c][1], b6 = dac[c][2];
            const uint32_t key = uint32_t(r6) | (uint32_t(g6) << 6) | (uint32_t(b6) << 12);
            size_t found = 0;
            while (found < palette_out->size() && seen_keys[found] != key) found++;
            if (found == palette_out->size()) {
                if (palette_out->size() >= 256) return 0;
                seen_keys[found] = key;
                palette_out->push_back({r6, g6, b6});
            }
            idx = int16_t(found);
            map[c] = idx;
        }
        mix(uint8_t(idx));
    }
    if (dac_to_canon) *dac_to_canon = map;
    return h;
}

uint64_t image_key_of(const std::vector<uint8_t>& pixels, int width, const std::array<std::array<uint8_t, 3>, 256>& dac) {
    uint64_t id = kFnvBasis;
    auto mix = [&id](uint8_t v) { id = (id ^ v) * kFnvPrime; };
    mix(0xFF);  // whole pictures never share a key with sprites
    mix(uint8_t(width));
    mix(uint8_t(width >> 8));
    for (uint8_t c : pixels) {
        mix(dac[c][0]);
        mix(dac[c][1]);
        mix(dac[c][2]);
    }
    return id;
}

// True when every colour entry is within +-2 VGA DAC units (out of 63) of the reference palette, covering
// fixed-point rounding drift across Utopia's 32-step palette fades without collapsing all-black silhouettes
// or distinct sprites whose palettes happen to be identical.
bool match_near_exact(const std::vector<std::array<uint8_t, 3>>& ref, const std::vector<std::array<uint8_t, 3>>& cur) {
    if (ref.size() != cur.size() || ref.empty()) return false;
    bool has_visible = false;
    int max_diff = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (std::max({int(ref[i][0]), int(ref[i][1]), int(ref[i][2])}) >= 6) has_visible = true;
        for (int ch = 0; ch < 3; ch++) {
            const int d = std::abs(int(cur[i][ch]) - int(ref[i][ch]));
            if (d > 2) return false;
            if (d > max_diff) max_diff = d;
        }
    }
    return has_visible && max_diff >= 1;
}

bool match_slot_near_exact(const Machine& m, const std::bitset<256>& colours, const std::array<std::array<uint8_t, 3>, 256>& ref_dac) {
    bool has_visible = false;
    for (int c = 0; c < 256; c++) {
        if (!colours[size_t(c)]) continue;
        const auto& r = ref_dac[size_t(c)];
        if (std::max({int(r[0]), int(r[1]), int(r[2])}) >= 6) has_visible = true;
        for (int ch = 0; ch < 3; ch++)
            if (std::abs(int(m.dac[c][ch]) - int(r[size_t(ch)])) > 2) return false;
    }
    return has_visible;
}

// Checks whether `cur` is a uniform ambient dimming/tinting (such as nighttime or winter weather) of `ref`.
// Rejects single-channel edits, brightening, fade-to-black (< 25%), single-colour blinks, and cross-planet hue swaps.
bool match_ambient_dim(const std::vector<std::array<uint8_t, 3>>& ref, const std::vector<std::array<uint8_t, 3>>& cur,
                       float tint_out[3], float* err_out = nullptr, const float* hint = nullptr) {
    if (ref.size() != cur.size() || ref.empty()) return false;
    for (size_t i = 0; i < ref.size(); i++)
        for (int ch = 0; ch < 3; ch++)
            if (int(cur[i][ch]) > int(ref[i][ch]) + 2) return false;
    auto try_fit = [&](bool allow_emissive) -> bool {
        double sum_ref[3] = {0, 0, 0}, sum_cur[3] = {0, 0, 0};
        int dimmed_entries = 0, unchanged_entries = 0, decreased_entries = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            const auto& r = ref[i];
            const auto& c = cur[i];
            const int rmax = std::max({int(r[0]), int(r[1]), int(r[2])});
            if (rmax < 6) continue;
            if (c[0] < r[0] || c[1] < r[1] || c[2] < r[2]) decreased_entries++;
            const bool same = allow_emissive && rmax >= 16 && std::abs(int(c[0]) - int(r[0])) <= 2 &&
                              std::abs(int(c[1]) - int(r[1])) <= 2 && std::abs(int(c[2]) - int(r[2])) <= 2;
            if (same) {
                unchanged_entries++;
                continue;
            }
            dimmed_entries++;
            for (int ch = 0; ch < 3; ch++) {
                sum_ref[ch] += r[ch];
                sum_cur[ch] += c[ch];
            }
        }
        if (dimmed_entries == 0) return false;
        // Require at least two entries to have actually decreased on 4+ colour sprites so a single mildly
        // blinking indicator on a daytime screen cannot trigger uniform dimming via the all-entries pass.
        if (!allow_emissive && ref.size() >= 4 && decreased_entries < 2) return false;
        // When emissive exceptions are allowed, dimmed colours must form a strict majority (and at least two
        // entries) so a single blinking indicator on a daytime screen never darkens the entire frame.
        if (allow_emissive && (dimmed_entries < 2 || dimmed_entries <= unchanged_entries)) return false;
        float s[3] = {-1.0f, -1.0f, -1.0f};
        int active = 0, dimmed_channels = 0;
        float avg = 0.0f;
        for (int ch = 0; ch < 3; ch++) {
            if (sum_ref[ch] >= 8.0) {
                s[ch] = float(sum_cur[ch] / sum_ref[ch]);
                active++;
                if (s[ch] < 0.96f) dimmed_channels++;
                avg += s[ch];
            }
        }
        if (active == 0) return false;
        avg /= float(active);
        for (int ch = 0; ch < 3; ch++)
            if (s[ch] < 0.0f) s[ch] = hint ? hint[ch] : avg;
        const float min_s = std::min({s[0], s[1], s[2]});
        const float max_s = std::max({s[0], s[1], s[2]});
        if (min_s < 0.25f || max_s > 0.98f) return false;
        if (max_s / min_s > 1.60f) return false;
        if (active >= 2 && dimmed_channels < 2) return false;
        if (active == 1 && !hint) return false;
        if (hint) {
            for (int ch = 0; ch < 3; ch++)
                if (std::fabs(s[ch] - hint[ch]) > 0.22f) return false;
        }
        float total_err = 0.0f;
        for (size_t i = 0; i < ref.size(); i++) {
            const auto& r = ref[i];
            const auto& c = cur[i];
            const int rmax = std::max({int(r[0]), int(r[1]), int(r[2])});
            if (rmax < 6) continue;
            const bool same = allow_emissive && rmax >= 16 && std::abs(int(c[0]) - int(r[0])) <= 2 &&
                              std::abs(int(c[1]) - int(r[1])) <= 2 && std::abs(int(c[2]) - int(r[2])) <= 2;
            if (same) continue;
            for (int ch = 0; ch < 3; ch++) {
                const float diff = std::fabs(float(c[ch]) - float(r[ch]) * s[ch]);
                if (diff > 3.5f) return false;
                total_err += diff;
            }
        }
        tint_out[0] = std::clamp(s[0], 0.25f, 1.0f);
        tint_out[1] = std::clamp(s[1], 0.25f, 1.0f);
        tint_out[2] = std::clamp(s[2], 0.25f, 1.0f);
        if (err_out) *err_out = total_err / float(dimmed_entries * 3);
        return true;
    };
    // Fit all visible entries first so mild uniform fades (92% to 97%, where mid-tones shift by only 1 or 2
    // DAC steps) never fall into a dead zone between match_near_exact and the emissive-window fallback.
    if (try_fit(false)) return true;
    if (ref.size() >= 4 && try_fit(true)) return true;
    // Allow at most one non-uniformly dimmed accent entry (for example a solar panel highlight dimming from
    // bright yellow to warm amber at night) when every remaining visible colour fits a tight uniform tint.
    if (ref.size() >= 6) {
        for (size_t skip = 0; skip < ref.size(); skip++) {
            const auto& sr = ref[skip];
            const auto& sc = cur[skip];
            if (std::max({int(sr[0]), int(sr[1]), int(sr[2])}) < 6) continue;
            bool valid_outlier = true;
            for (int ch = 0; ch < 3; ch++) {
                if (int(sc[ch]) > int(sr[ch]) || (sr[ch] == 0) != (sc[ch] == 0) ||
                    float(sc[ch]) < float(sr[ch]) * 0.25f - 2.0f) {
                    valid_outlier = false;
                    break;
                }
            }
            if (!valid_outlier) continue;
            double sum_ref[3] = {0, 0, 0}, sum_cur[3] = {0, 0, 0};
            int dimmed_entries = 0, decreased_entries = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                if (i == skip) continue;
                const auto& r = ref[i];
                const auto& c = cur[i];
                if (std::max({int(r[0]), int(r[1]), int(r[2])}) < 6) continue;
                if (c[0] < r[0] || c[1] < r[1] || c[2] < r[2]) decreased_entries++;
                dimmed_entries++;
                for (int ch = 0; ch < 3; ch++) {
                    sum_ref[ch] += r[ch];
                    sum_cur[ch] += c[ch];
                }
            }
            if (dimmed_entries < 5 || decreased_entries < 4) continue;
            float s[3] = {-1.0f, -1.0f, -1.0f};
            int active = 0, dimmed_channels = 0;
            float avg = 0.0f;
            for (int ch = 0; ch < 3; ch++) {
                if (sum_ref[ch] >= 8.0) {
                    s[ch] = float(sum_cur[ch] / sum_ref[ch]);
                    active++;
                    if (s[ch] < 0.96f) dimmed_channels++;
                    avg += s[ch];
                }
            }
            if (active < 2 || dimmed_channels < 2) continue;
            avg /= float(active);
            for (int ch = 0; ch < 3; ch++)
                if (s[ch] < 0.0f) s[ch] = hint ? hint[ch] : avg;
            const float min_s = std::min({s[0], s[1], s[2]});
            const float max_s = std::max({s[0], s[1], s[2]});
            if (min_s < 0.25f || max_s > 0.98f || max_s / min_s > 1.60f) continue;
            if (hint && (std::fabs(s[0] - hint[0]) > 0.22f || std::fabs(s[1] - hint[1]) > 0.22f || std::fabs(s[2] - hint[2]) > 0.22f))
                continue;
            float total_err = 0.0f;
            bool ok = true;
            for (size_t i = 0; i < ref.size(); i++) {
                if (i == skip) continue;
                const auto& r = ref[i];
                const auto& c = cur[i];
                if (std::max({int(r[0]), int(r[1]), int(r[2])}) < 6) continue;
                for (int ch = 0; ch < 3; ch++) {
                    const float diff = std::fabs(float(c[ch]) - float(r[ch]) * s[ch]);
                    if (diff > 3.5f) {
                        ok = false;
                        break;
                    }
                    total_err += diff;
                }
                if (!ok) break;
            }
            if (!ok) continue;
            tint_out[0] = std::clamp(s[0], 0.25f, 1.0f);
            tint_out[1] = std::clamp(s[1], 0.25f, 1.0f);
            tint_out[2] = std::clamp(s[2], 0.25f, 1.0f);
            if (err_out) *err_out = total_err / float(dimmed_entries * 3);
            return true;
        }
    }
    return false;
}

bool match_slot_ambient(const Machine& m, const std::bitset<256>& colours, const std::array<std::array<uint8_t, 3>, 256>& ref_dac,
                        float tint_out[3], const float* hint = nullptr) {
    std::vector<std::array<uint8_t, 3>> ref, cur;
    for (int c = 0; c < 256; c++) {
        if (!colours[size_t(c)]) continue;
        ref.push_back(ref_dac[size_t(c)]);
        cur.push_back({m.dac[c][0], m.dac[c][1], m.dac[c][2]});
    }
    return match_ambient_dim(ref, cur, tint_out, nullptr, hint);
}

}  // namespace

HdSprites::HdSprites()
    : sprite_marks_(0x100000),
      image_marks_(0x100000),
      front_sprite_marks_(kPixels),
      front_image_marks_(kPixels),
      stable_vram_(kPixels, 0),
      vram_masked_(kPixels, 0) {}

void HdSprites::index_disk_shapes() {
    disk_shapes_.clear();
    disk_images_.clear();
    disk_shapes_version_++;
    if (folder_.empty()) return;
    std::error_code ec;
    const auto orig_dir = folder_ / L"original";
    if (!std::filesystem::exists(orig_dir, ec)) return;
    for (const auto& entry : std::filesystem::directory_iterator(orig_dir, ec)) {
        if (ec) break;
        std::error_code file_ec;
        if (!entry.is_regular_file(file_ec) || file_ec) continue;
        const std::wstring stem = entry.path().stem().wstring();
        if (stem.size() != 16 || entry.path().extension() != L".png") continue;
        unsigned long long key = 0;
        if (swscanf(stem.c_str(), L"%llx", &key) != 1) continue;
        unsigned w = 0, h = 0;
        std::vector<uint32_t> rgba;
        if (!read_uncompressed_png(entry.path(), &w, &h, &rgba)) continue;
        const bool has_up = std::filesystem::exists(upscaled_file(key), file_ec) && !file_ec;
        if (w > 256 || h > 256) {
            // Index upscaled full-screen backdrops so fade-in and near-exact palette transitions switch to HD immediately.
            if (!has_up || w > 1024 || h > 1024) continue;
            std::vector<std::array<uint8_t, 3>> pal;
            const uint64_t shape = canonical_image_shape_rgba(rgba, int(w), int(h), &pal);
            if (shape && !pal.empty()) disk_images_[shape].push_back({key, int(w), int(h), true, std::move(pal)});
            continue;
        }
        std::vector<std::array<uint8_t, 3>> pal;
        const uint64_t shape = canonical_shape(rgba, int(w), int(h), &pal);
        if (!shape || pal.empty()) continue;
        disk_shapes_[shape].push_back({key, int(w), int(h), has_up, std::move(pal)});
    }
}

void HdSprites::set_folder(const std::filesystem::path& folder) {
    std::lock_guard<std::mutex> lock(mutex_);
    folder_ = folder;
    std::error_code ec;
    std::filesystem::create_directories(folder / L"original", ec);
    std::filesystem::create_directories(folder / L"upscaled", ec);
    index_disk_shapes();
}

std::filesystem::path HdSprites::original_file(uint64_t key) const {
    wchar_t name[32];
    swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(key));
    return folder_ / L"original" / name;
}

std::filesystem::path HdSprites::upscaled_file(uint64_t key) const {
    wchar_t name[32];
    swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(key));
    return folder_ / L"upscaled" / name;
}

bool HdSprites::settled(const std::bitset<256>& colours) const {
    const int64_t now = now_ms();
    for (int c = 0; c < 256; c++)
        if (colours[size_t(c)] && now - colour_changed_[c].load() < kSettleMs) return false;
    return true;
}

// The sprites' pairs in the layout file under their new names, once their pictures have moved to them.
void HdSprites::rename_in_layout(const std::filesystem::path& file, uint64_t from, uint64_t to) {
    std::vector<NeighbourLog::Pair> pairs;
    FILE* f = _wfopen(file.c_str(), L"rbN");
    if (!f) return;
    char line[128];
    bool changed = false;
    while (fgets(line, sizeof line, f)) {
        unsigned long long a, b;
        int dx, dy;
        if (sscanf(line, "%llx %llx %d %d", &a, &b, &dx, &dy) != 4) continue;
        NeighbourLog::Pair p{a, b, dx, dy};
        if (p.a == from || p.b == from) {
            changed = true;
            if (p.a == from) p.a = to;
            if (p.b == from) p.b = to;
            if (p.a > p.b) p = {p.b, p.a, -p.dx, -p.dy};
        }
        pairs.push_back(p);
    }
    fclose(f);
    if (!changed) return;
    // Written next to the file and moved over it, so an interrupted write leaves the old file.
    const std::filesystem::path temporary = file.wstring() + L".part";
    FILE* out = _wfopen(temporary.c_str(), L"wbN");
    if (!out) return;
    bool ok = true;
    for (const auto& p : pairs)
        ok = ok && fprintf(out, "%016llx %016llx %d %d\n", static_cast<unsigned long long>(p.a), static_cast<unsigned long long>(p.b), p.dx, p.dy) > 0;
    ok = fclose(out) == 0 && ok;
    if (!ok || !MoveFileExW(temporary.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ec;
        std::filesystem::remove(temporary, ec);
        return;
    }
    // Pairs that were loaded already are known under their new names too, so they are not written again.
    if (layout_loaded_)
        for (const auto& p : pairs) neighbours_.known(p);
}

// Called without the lock held: the lock is taken only to look the sprite up and to add it, and the picture is
// saved in between, so neither thread waits for a file to be written.
uint32_t HdSprites::slot_for(uint64_t id, uint64_t legacy, int width, int rows, const std::bitset<256>& colours, uint64_t palette,
                             const std::array<std::array<uint8_t, 3>, 256>& ref_dac,
                             const std::function<std::vector<uint32_t>()>& picture) {
    std::filesystem::path original, upscaled, old_original, old_upscaled, layout;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_key_.find(id);
        if (it != by_key_.end()) return it->second;
        if (!folder_.empty()) {
            original = original_file(id);
            upscaled = upscaled_file(id);
            old_original = original_file(legacy);
            old_upscaled = upscaled_file(legacy);
            layout = folder_ / L"layout.txt";
        }
    }
    State state = State::missing;
    if (!original.empty()) {
        std::error_code ec;
        // A sprite saved under its earlier name keeps its pictures under the new one. The upscaled picture is
        // also looked for later (see take_pending), in case it is written under the earlier name afterwards or
        // cannot be moved now.
        if (legacy != id) {
            moved(old_upscaled, upscaled);
            if (!std::filesystem::exists(original, ec) && moved(old_original, original)) rename_in_layout(layout, legacy, id);
        }
        if (!std::filesystem::exists(original, ec)) write_png(original, picture().data(), unsigned(width), unsigned(rows), 1, true);
        if (std::filesystem::exists(upscaled, ec)) state = State::unloaded;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = by_key_.find(id);
    if (it != by_key_.end()) return it->second;
    const uint32_t slot = uint32_t(slots_.size());
    slots_.push_back({id, legacy, width, rows, state, 0, colours, palette, ref_dac, true});
    by_key_.emplace(id, slot);
    return slot;
}

void HdSprites::draw(const Machine& m, uint32_t data, int width, int rows, uint32_t dest, uint8_t base, bool bottom_up, bool track_neighbours) {
    if (width <= 0 || width > 256 || rows <= 0 || rows > 256) return;
    const uint32_t stride = uint32_t(width / 2);
    // Only the palette entries the sprite's pixels actually use count, for its key, for waiting while colours
    // fade and for when its upscaled picture gives way to the original pixels: games often let another colour
    // of the same bank blink or cycle (a lamp, a cursor), and that must not make every sprite of the bank
    // flicker back to its pixels or be saved again for every state of that colour.
    std::bitset<256> used;
    for (uint32_t i = 0; i < stride * uint32_t(rows); i++) {
        const uint8_t byte = m.mem[(data + i) & 0xFFFFF];
        if (byte >> 4) used.set(uint8_t(base + (byte >> 4)));
        if (byte & 0x0F) used.set(uint8_t(base + (byte & 0x0F)));
    }
    // While the colours fade, unknown sprites wait until they settle so passing shades are not saved to disk.
    // Sprites that already match a known upscaled asset still mark immediately so terrain and buildings appear
    // without a multi-second delay after loading a game or leaving a menu.
    const bool is_settled = settled(used);
    // The key covers the pixels and the colours they are drawn in, so a sprite shown with another palette
    // (another planet, for example) is a sprite of its own.
    uint64_t id = kFnvBasis;
    auto mix = [](uint64_t& h, uint8_t v) { h = (h ^ v) * kFnvPrime; };
    if (!bottom_up) mix(id, 0xFE);  // top-down sprites never collide with bottom-up sprites of the same bytes
    for (uint32_t i = 0; i < stride * uint32_t(rows); i++) mix(id, m.mem[(data + i) & 0xFFFFF]);
    mix(id, uint8_t(width));
    // Earlier versions named bottom-up sprites after all 15 colours of the bank; their pictures are taken over.
    uint64_t legacy = id;
    for (int c = 1; c < 16; c++)
        for (int k = 0; k < 3; k++) {
            const uint8_t v = m.dac[uint8_t(base + c)][k];
            if (bottom_up) mix(legacy, v);
            if (used[uint8_t(base + c)]) mix(id, v);
        }
    if (!bottom_up) legacy = id;

    auto build_rgba = [&] {
        // Stored top row first, as pictures are, with transparent pixels where colour 0 is.
        std::vector<uint32_t> rgba(size_t(width) * size_t(rows));
        for (int y = 0; y < rows; y++) {
            const uint32_t row = data + uint32_t(bottom_up ? rows - 1 - y : y) * stride;
            for (int x = 0; x < width; x++) {
                const uint8_t c = nibble(m, row, x);
                rgba[size_t(y) * size_t(width) + size_t(x)] = c ? colour_of(m, uint8_t(base + c)) : 0;
            }
        }
        return rgba;
    };

    auto ref_dac = copy_dac(m);
    uint64_t slot_palette = palette_of(m, used);
    // If this exact (pixels + palette) key is not yet known and not upscaled on disk, check whether this
    // sprite is a near-exact (+-2 DAC rounding) or uniformly dimmed nighttime/weather version of an existing
    // daytime sprite on disk (or one discovered earlier in this session).
    bool matched_daytime = false;
    bool known_exact = false;
    bool known_upscaled = false;
    bool known_missing = false;
    bool have_hint = false;
    uint32_t exact_slot = 0;
    uint64_t shapes_ver = 0;
    bool need_recheck_missing = false;
    uint64_t computed_shape = 0;
    std::vector<std::array<uint8_t, 3>> computed_pal;
    {
        bool has_candidates = false;
        std::filesystem::path exact_up;
        float hint[3] = {1.0f, 1.0f, 1.0f};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shapes_ver = disk_shapes_version_;
            if (ambient_tint_[0] < 0.96f || ambient_tint_[1] < 0.96f || ambient_tint_[2] < 0.96f) {
                hint[0] = ambient_tint_[0];
                hint[1] = ambient_tint_[1];
                hint[2] = ambient_tint_[2];
                have_hint = true;
            }
            auto kit = by_key_.find(id);
            if (kit != by_key_.end()) {
                known_exact = true;
                exact_slot = kit->second;
                const Slot& es = slots_[exact_slot];
                known_upscaled = es.state == State::unloaded || es.state == State::pending || es.state == State::placed;
                known_missing = es.state == State::missing;
                if (known_missing) {
                    const bool hint_shifted =
                        have_hint && (!es.miss_with_hint || std::fabs(hint[0] - es.miss_hint[0]) > 0.05f ||
                                      std::fabs(hint[1] - es.miss_hint[1]) > 0.05f || std::fabs(hint[2] - es.miss_hint[2]) > 0.05f);
                    need_recheck_missing = (es.miss_shape_version != shapes_ver) || hint_shifted;
                }
            }
            has_candidates = !disk_shapes_.empty();
            if (!known_exact && !folder_.empty()) exact_up = upscaled_file(id);
        }
        std::error_code ec;
        const bool exact_on_disk = !known_exact && !exact_up.empty() && std::filesystem::exists(exact_up, ec);
        if (exact_on_disk) known_upscaled = true;
        // Look up shape matches for unknown sprites, and recheck known missing slots only when disk_shapes_
        // gained new entries or an ambient dim hint newly became available, avoiding per-frame shape hashing.
        const bool search_shapes = (!known_exact && !exact_on_disk) || (known_missing && has_candidates && need_recheck_missing);
        if (search_shapes) {
            const std::vector<uint32_t> rgba = build_rgba();
            computed_shape = canonical_shape(rgba, width, rows, &computed_pal);
            if (has_candidates && computed_shape && !computed_pal.empty()) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto sit = disk_shapes_.find(computed_shape);
                if (sit != disk_shapes_.end()) {
                    auto find_best = [&](bool require_upscaled, bool near_exact_only, const float* use_hint) -> const DiskShape* {
                        const DiskShape* best = nullptr;
                        float best_err = 1e9f;
                        for (const auto& cand : sit->second) {
                            if (cand.width != width || cand.rows != rows) continue;
                            if (require_upscaled && !cand.has_upscaled) continue;
                            float err = 0.0f;
                            if (near_exact_only) {
                                if (!match_near_exact(cand.colours, computed_pal)) continue;
                            } else {
                                float tint[3];
                                if (!match_ambient_dim(cand.colours, computed_pal, tint, &err, use_hint)) continue;
                            }
                            const bool better = !best || (cand.has_upscaled && !best->has_upscaled) ||
                                                (cand.has_upscaled == best->has_upscaled && err < best_err);
                            if (better) {
                                best = &cand;
                                best_err = err;
                            }
                        }
                        return best;
                    };
                    // Always prefer an already-upscaled daytime candidate (near-exact, then hintless ambient
                    // dim, then hinted ambient dim) so stale un-upscaled nighttime PNGs never shadow HD assets.
                    const DiskShape* best = find_best(true, true, nullptr);
                    if (!best) best = find_best(true, false, nullptr);
                    if (!best && have_hint) best = find_best(true, false, hint);
                    if (!best && !known_exact && is_settled) {
                        best = find_best(false, true, nullptr);
                        if (!best) best = find_best(false, false, nullptr);
                        if (!best && have_hint) best = find_best(false, false, hint);
                    }
                    if (best) {
                        id = best->key;
                        legacy = best->key;
                        matched_daytime = true;
                        if (best->has_upscaled) known_upscaled = true;
                        // Reconstruct the sprite's daytime reference DAC for its used palette entries so
                        // frame() keeps matching both now (dimmed) and later when full daylight returns.
                        std::vector<uint32_t> seen_rgb;
                        for (int y = 0; y < rows; y++) {
                            const uint32_t row = data + uint32_t(bottom_up ? rows - 1 - y : y) * stride;
                            for (int x = 0; x < width; x++) {
                                const uint8_t c = nibble(m, row, x);
                                if (!c) continue;
                                const uint32_t rgb = colour_of(m, uint8_t(base + c)) & 0x00FFFFFFu;
                                size_t ci = 0;
                                while (ci < seen_rgb.size() && seen_rgb[ci] != rgb) ci++;
                                if (ci == seen_rgb.size()) seen_rgb.push_back(rgb);
                                if (ci < best->colours.size()) ref_dac[size_t(uint8_t(base + c))] = best->colours[ci];
                            }
                        }
                        slot_palette = palette_of_dac(ref_dac, used);
                    }
                }
            }
            if (known_missing && !matched_daytime) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (exact_slot < slots_.size()) {
                    slots_[exact_slot].miss_shape_version = shapes_ver;
                    slots_[exact_slot].miss_with_hint = have_hint;
                    slots_[exact_slot].miss_hint[0] = hint[0];
                    slots_[exact_slot].miss_hint[1] = hint[1];
                    slots_[exact_slot].miss_hint[2] = hint[2];
                }
            }
        }
    }
    if (!is_settled && !known_upscaled) return;

    const uint32_t slot = slot_for(id, legacy, width, rows, used, slot_palette, ref_dac, build_rgba);
    // Register newly discovered sprites' shapes so nightfall or newly finished background upscales can match them.
    if (!known_exact && !matched_daytime && width <= 256 && rows <= 256) {
        if (!computed_shape && computed_pal.empty()) {
            const std::vector<uint32_t> rgba = build_rgba();
            computed_shape = canonical_shape(rgba, width, rows, &computed_pal);
        }
        if (computed_shape && !computed_pal.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            slots_[slot].shape = computed_shape;
            if (!have_hint) {
                auto& list = disk_shapes_[computed_shape];
                const bool exists = std::any_of(list.begin(), list.end(), [&](const DiskShape& d) { return d.key == id; });
                if (!exists) {
                    const bool has_up = slots_[slot].state == State::unloaded || slots_[slot].state == State::pending ||
                                        slots_[slot].state == State::placed;
                    list.push_back({id, width, rows, has_up, std::move(computed_pal)});
                    disk_shapes_version_++;
                }
            }
        }
    }
    if (track_neighbours && bottom_up) record_neighbours(id, width, rows, dest);
    // Sprites drawn later lie on top. The count runs from 1 to 2046 and starts again; 0 marks pictures.
    const uint16_t order = uint16_t(draw_count_ % 2046 + 1);
    draw_count_++;

    bool touched_vram = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int r = 0; r < rows; r++) {
            const uint32_t row = data + uint32_t(r) * stride;
            const uint32_t line = bottom_up ? dest - uint32_t(r) * 320 : dest + uint32_t(r) * 320;
            const uint16_t py = uint16_t(bottom_up ? rows - 1 - r : r);
            for (int x = 0; x < width; x++) {
                const uint8_t c = nibble(m, row, x);
                if (!c) continue;
                const uint32_t pos = (line + uint32_t(x)) & 0xFFFFF;
                sprite_marks_[pos] = {slot + 1, uint16_t(x), py, uint8_t(base + c), order};
                if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) touched_vram = true;
            }
        }
        if (touched_vram) vram_dirty_.store(true, std::memory_order_release);
    }
}

void HdSprites::record_neighbours(uint64_t key, int width, int rows, uint32_t dest) {
    std::filesystem::path file;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (folder_.empty()) return;
        file = folder_ / L"layout.txt";
    }
    if (!layout_loaded_) {
        // Pairs recorded in earlier runs are not written again.
        layout_loaded_ = true;
        // Read line by line, so a line cut short by a crash is skipped and the lines after it still count.
        if (FILE* f = _wfopen(file.c_str(), L"rbN")) {
            char line[128];
            while (fgets(line, sizeof line, f)) {
                unsigned long long a, b;
                int dx, dy;
                if (sscanf(line, "%llx %llx %d %d", &a, &b, &dx, &dy) == 4) neighbours_.known({a, b, dx, dy});
            }
            fclose(f);
        }
    }
    const auto pairs = neighbours_.drawn(key, width, rows, dest, frames_.load());
    if (pairs.empty()) return;
    if (FILE* f = _wfopen(file.c_str(), L"abN")) {
        for (const auto& p : pairs)
            fprintf(f, "%016llx %016llx %d %d\n", static_cast<unsigned long long>(p.a), static_cast<unsigned long long>(p.b), p.dx, p.dy);
        fclose(f);
    }
}

void NeighbourLog::known(const Pair& pair) { reported_.insert(pair); }

std::vector<NeighbourLog::Pair> NeighbourLog::drawn(uint64_t key, int width, int rows, uint32_t dest, uint32_t frame) {
    std::vector<Pair> out;
    for (size_t i = 0; i < recent_count_; i++) {
        const Recent& r = recent_[i];
        // The pieces of one building can differ in height, but tiles of one grid have the same width. A flat
        // tile (at most half as high as wide) and a tall sprite are never neighbours, so buildings and units
        // over the ground are left out.
        if (r.key == key || frame - r.frame > 1 || r.width != width || (r.rows <= width / 2) != (rows <= width / 2)) continue;
        // Where this sprite lies from the earlier one, from the distance of their bottom rows in memory: whole
        // lines of 320 pixels down, and the rest across, between 160 to the left and 159 to the right.
        const int64_t d = int64_t(dest) - int64_t(r.dest);
        if (d <= -kPixels || d >= kPixels) continue;
        const int64_t n = d + 160;
        const int dy = int(n >= 0 ? n / 320 : -((-n + 319) / 320));
        const int dx = int(d - int64_t(dy) * 320);
        // Only tiles on an isometric grid count: half a tile across and a quarter of its width up or down, for
        // flat tiles and for the pieces of a building that stands on several tiles alike. Units standing on a
        // tile and buildings over the ground are not neighbours.
        const int half_x = width / 2, half_y = width / 4;
        if ((dx != half_x && dx != -half_x) || (dy != half_y && dy != -half_y)) continue;
        const Pair p = r.key < key ? Pair{r.key, key, dx, dy} : Pair{key, r.key, -dx, -dy};
        if (reported_.count(p)) continue;
        Seen& s = seen_[p];
        if (s.count == 0 || s.frame != frame) {
            s.count++;
            s.frame = frame;
        }
        if (s.count >= 3) {
            seen_.erase(p);
            reported_.insert(p);
            out.push_back(p);
        }
    }
    recent_[recent_next_] = {key, width, rows, dest, frame};
    recent_next_ = (recent_next_ + 1) % kRecent;
    recent_count_ = std::min(recent_count_ + 1, kRecent);
    // Pairs that were only seen once or twice, after scrolling for example, are forgotten now and then.
    if (seen_.size() > 10000) seen_.clear();
    return out;
}

void HdSprites::image(const Machine& m, uint32_t data, int width, int rows, uint32_t stride, uint32_t dest, uint32_t dest_stride) {
    if (width <= 0 || width > 1024 || rows <= 0 || rows > 1024) return;
    Deferred d{std::vector<uint8_t>(size_t(width) * size_t(rows)), width, rows, dest, dest_stride, 0};
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < width; x++) d.pixels[size_t(y) * size_t(width) + size_t(x)] = m.mem[(data + uint32_t(y) * stride + uint32_t(x)) & 0xFFFFF];
    std::lock_guard<std::mutex> lock(mutex_);
    d.epoch = vram_epoch_;
    d.seq = ++deferred_seq_;
    d.present_seq = present_seq_;
    // A newly unpacked picture covers any sprites or older picture previously drawn in the same region.
    for (int y = 0; y < rows; y++) {
        const uint32_t row_base = dest + uint32_t(y) * dest_stride;
        for (int x = 0; x < width; x++) {
            const uint32_t pos = (row_base + uint32_t(x)) & 0xFFFFF;
            sprite_marks_[pos] = {};
            image_marks_[pos] = {};
            if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) vram_masked_[pos - kVideo] = 0;
        }
    }
    // A waiting picture that the new one covers completely would never be seen, so it is dropped. Beyond that
    // at most kMaxDeferred pictures wait, and only then the oldest gives way.
    auto covered = [&d](const Deferred& old) {
        if (old.dest_stride != d.dest_stride || !d.dest_stride) return false;
        const int64_t ox = old.dest % old.dest_stride, oy = old.dest / old.dest_stride;
        const int64_t nx = d.dest % d.dest_stride, ny = d.dest / d.dest_stride;
        return ox >= nx && oy >= ny && ox + old.width <= nx + d.width && oy + old.rows <= ny + d.rows;
    };
    std::erase_if(deferred_, covered);
    if (deferred_.size() >= kMaxDeferred) deferred_.erase(deferred_.begin());
    deferred_.push_back(std::move(d));
}

void HdSprites::take_over(const Machine& m, const Deferred& d, uint64_t matched_key,
                          const std::array<std::array<uint8_t, 3>, 256>* matched_dac) {
    std::bitset<256> colours;
    for (uint8_t c : d.pixels) colours.set(c);
    const auto cur_dac = copy_dac(m);
    const uint64_t id = matched_key ? matched_key : image_key_of(d.pixels, d.width, cur_dac);
    const auto ref_dac = matched_dac ? *matched_dac : cur_dac;
    const uint64_t pal_hash = matched_dac ? palette_of_dac(ref_dac, colours) : palette_of_dac(cur_dac, colours);
    const uint32_t slot = slot_for(id, id, d.width, d.rows, colours, pal_hash, ref_dac, [&] {
        std::vector<uint32_t> rgba(d.pixels.size());
        for (size_t i = 0; i < rgba.size(); i++) rgba[i] = colour_of(m, d.pixels[i]);
        return rgba;
    });
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(deferred_.begin(), deferred_.end(), [&](const Deferred& item) { return item.seq == d.seq; });
    if (it == deferred_.end()) return;  // superseded by a newer picture or present_buffer() while settling
    const Deferred latest = std::move(*it);
    deferred_.erase(it);
    // If VRAM overlays were cleared or the game switched back to its offscreen buffer while a direct-VRAM
    // picture was waiting to settle, skip marking VRAM so the closed screen cannot reappear over the live frame.
    const bool dest_is_vram = latest.dest >= kVideo && latest.dest < kVideo + uint32_t(kPixels);
    if (dest_is_vram && (latest.epoch != vram_epoch_ || screen_source_.load(std::memory_order_acquire) != kVideo)) return;
    bool touched_vram = false;
    // Mirror into front_image_marks_ only when present_buffer() already presented this unpacked picture to VRAM,
    // handling both full-screen and partial-row unpacks into the offscreen buffer.
    const bool mirror_front = has_front_ && present_seq_ > latest.present_seq && latest.dest_stride == 320 &&
                              latest.dest >= front_source_ && latest.dest < front_source_ + uint32_t(kPixels);
    for (int y = 0; y < latest.rows; y++)
        for (int x = 0; x < latest.width; x++) {
            const uint32_t pos = (latest.dest + uint32_t(y) * latest.dest_stride + uint32_t(x)) & 0xFFFFF;
            const size_t idx = size_t(y) * size_t(latest.width) + size_t(x);
            const uint8_t colour = latest.expected.empty() ? latest.pixels[idx] : latest.expected[idx];
            const bool masked = !latest.masked.empty() && latest.masked[idx] != 0;
            const Mark mark{(slot + 1) | (masked ? 0x80000000u : 0u), uint16_t(x), uint16_t(y), colour, 0};
            image_marks_[pos] = mark;
            if (mirror_front && pos >= front_source_ && pos < front_source_ + uint32_t(kPixels))
                front_image_marks_[pos - front_source_] = mark;
            if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) touched_vram = true;
        }
    if (touched_vram) vram_dirty_.store(true, std::memory_order_release);
}

void HdSprites::set_screen_source(uint32_t linear) { screen_source_.store(linear, std::memory_order_release); }

void HdSprites::begin_present(const Machine& m) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::memcpy(stable_vram_.data(), m.vram(), size_t(kPixels));
    presenting_ = true;
}

void HdSprites::present_buffer(uint32_t linear) {
    std::lock_guard<std::mutex> lock(mutex_);
    screen_source_.store(linear, std::memory_order_release);
    presenting_ = false;
    if (linear == kVideo) {
        has_front_ = false;
        front_source_ = 0;
        return;
    }
    if (linear != 0 && linear <= 0x100000u - uint32_t(kPixels)) {
        std::copy_n(sprite_marks_.data() + linear, size_t(kPixels), front_sprite_marks_.data());
        std::copy_n(image_marks_.data() + linear, size_t(kPixels), front_image_marks_.data());
        has_front_ = true;
        front_source_ = linear;
        present_seq_++;
    } else {
        has_front_ = false;
        front_source_ = 0;
    }
    vram_epoch_++;
    std::erase_if(deferred_, [](const Deferred& d) { return d.dest >= kVideo && d.dest < kVideo + uint32_t(kPixels); });
    if (vram_dirty_.exchange(false, std::memory_order_acq_rel)) {
        std::fill_n(sprite_marks_.data() + kVideo, kPixels, Mark{});
        std::fill_n(image_marks_.data() + kVideo, kPixels, Mark{});
        std::fill(vram_masked_.begin(), vram_masked_.end(), uint8_t(0));
    }
}

void HdSprites::clear_rect(uint32_t dest, int width, int rows) {
    if (width <= 0 || width > 320 || rows <= 0 || rows > 200) return;
    std::lock_guard<std::mutex> lock(mutex_);
    bool touched_vram = false;
    for (int y = 0; y < rows; y++) {
        const uint32_t row = dest + uint32_t(y) * 320;
        for (int x = 0; x < width; x++) {
            const uint32_t pos = (row + uint32_t(x)) & 0xFFFFF;
            if (sprite_marks_[pos].slot) sprite_marks_[pos].slot |= 0x80000000u;
            image_marks_[pos].slot |= 0x80000000u;
            if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) {
                vram_masked_[pos - kVideo] = 2;  // 2 = genuine clear_rect popup mask (1 = sync_image_rect text clear)
                touched_vram = true;
            }
            for (auto& d : deferred_) {
                if (d.dest_stride != 320 || d.width <= 0 || d.rows <= 0 || pos < d.dest) continue;
                const uint32_t rel = pos - d.dest;
                const int dy = int(rel / 320), dx = int(rel % 320);
                if (dy >= d.rows || dx >= d.width) continue;
                if (d.masked.empty()) d.masked.assign(d.pixels.size(), 0);
                d.masked[size_t(dy) * size_t(d.width) + size_t(dx)] = 1;
            }
        }
    }
    if (touched_vram) vram_dirty_.store(true, std::memory_order_release);
}

void HdSprites::restore_rect(uint32_t dest, int width, int rows) {
    if (width <= 0 || width > 320 || rows <= 0 || rows > 200) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int y = 0; y < rows; y++) {
        const uint32_t row = dest + uint32_t(y) * 320;
        for (int x = 0; x < width; x++) {
            const uint32_t pos = (row + uint32_t(x)) & 0xFFFFF;
            sprite_marks_[pos].slot &= ~0x80000000u;
            image_marks_[pos].slot &= ~0x80000000u;
            if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) vram_masked_[pos - kVideo] = 0;
            for (auto& d : deferred_) {
                if (d.masked.empty() || d.dest_stride != 320 || d.width <= 0 || d.rows <= 0 || pos < d.dest) continue;
                const uint32_t rel = pos - d.dest;
                const int dy = int(rel / 320), dx = int(rel % 320);
                if (dy >= d.rows || dx >= d.width) continue;
                d.masked[size_t(dy) * size_t(d.width) + size_t(dx)] = 0;
            }
        }
    }
}

void HdSprites::restore_backdrop_rect(uint32_t dest, int width, int rows) {
    if (width <= 0 || width > 320 || rows <= 0 || rows > 200) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (int y = 0; y < rows; y++) {
        const uint32_t row = dest + uint32_t(y) * 320;
        for (int x = 0; x < width; x++) {
            const uint32_t pos = (row + uint32_t(x)) & 0xFFFFF;
            sprite_marks_[pos] = {};
            image_marks_[pos].slot &= ~0x80000000u;
            if (pos >= kVideo && pos < kVideo + uint32_t(kPixels)) vram_masked_[pos - kVideo] = 0;
            for (auto& d : deferred_) {
                if (d.masked.empty() || d.dest_stride != 320 || d.width <= 0 || d.rows <= 0 || pos < d.dest) continue;
                const uint32_t rel = pos - d.dest;
                const int dy = int(rel / 320), dx = int(rel % 320);
                if (dy >= d.rows || dx >= d.width) continue;
                d.masked[size_t(dy) * size_t(d.width) + size_t(dx)] = 0;
            }
        }
    }
}

void HdSprites::sync_image_rect(const Machine& m, uint32_t dest, int width, int rows) {
    if (width <= 0 || width > 320 || rows <= 0 || rows > 200) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const uint32_t source = screen_source_.load(std::memory_order_acquire);
    const bool use_front = has_front_ && source == front_source_;
    bool touched_vram = false;
    for (int y = 0; y < rows; y++) {
        const uint32_t row = dest + uint32_t(y) * 320;
        for (int x = 0; x < width; x++) {
            const uint32_t pos = (row + uint32_t(x)) & 0xFFFFF;
            const uint8_t val = m.mem[pos];
            const bool in_vram = pos >= kVideo && pos < kVideo + uint32_t(kPixels);
            // The opaque 8x8/8x6 clear tile overwrites every pixel in the cell, erasing any prior sprite there
            // (for example, buildings from an earlier page of the build menu when switching pages).
            sprite_marks_[pos] = {};
            // Update an already-active unmasked backdrop mark (either a direct-VRAM screen such as option.vgg
            // or a previous VRAM text-clear overlay) before skipping masked popup pixels.
            if (image_marks_[pos].slot && (image_marks_[pos].slot & 0x80000000u) == 0) {
                image_marks_[pos].colour = val;
            } else if (in_vram && !vram_masked_[pos - kVideo] && !image_marks_[pos].slot && source != 0 && source != kVideo) {
                // The text-clear wrote only to VRAM, not to the offscreen buffer, so record a VRAM overlay mark
                // instead of mutating the offscreen buffer's expected pixel colours.
                const uint32_t voff = pos - kVideo;
                const uint32_t src_pos = (source + voff) & 0xFFFFF;
                const Mark& src_img = use_front ? front_image_marks_[voff] : image_marks_[src_pos];
                const Mark& src_spr = use_front ? front_sprite_marks_[voff] : sprite_marks_[src_pos];
                if (src_img.slot && (src_img.slot & 0x80000000u) == 0 && !src_spr.slot) {
                    image_marks_[pos] = src_img;
                    image_marks_[pos].colour = val;
                    vram_masked_[voff] = 1;
                    touched_vram = true;
                }
            }
            if (in_vram && vram_masked_[pos - kVideo]) continue;
            for (auto& d : deferred_) {
                if (d.dest_stride != 320 || d.width <= 0 || d.rows <= 0) continue;
                const bool d_vram = d.dest >= kVideo && d.dest < kVideo + uint32_t(kPixels);
                if (d_vram != in_vram || pos < d.dest) continue;
                const uint32_t rel = pos - d.dest;
                const int dy = int(rel / 320), dx = int(rel % 320);
                if (dy >= d.rows || dx >= d.width) continue;
                if (!d.masked.empty() && d.masked[size_t(dy) * size_t(d.width) + size_t(dx)] != 0) continue;
                if (d.expected.empty()) d.expected = d.pixels;
                d.expected[size_t(dy) * size_t(d.width) + size_t(dx)] = val;
            }
        }
    }
    if (touched_vram) vram_dirty_.store(true, std::memory_order_release);
}

void HdSprites::frame(const Machine& m, uint32_t* refs, const uint8_t* vram_snapshot) {
    frame_impl(m, refs, vram_snapshot, nullptr);
}

void HdSprites::snapshot_and_frame(const Machine& m, uint8_t* vram_out, uint32_t* refs,
                                   const std::function<void(uint8_t*)>& preprocess_vram) {
    frame_impl(m, refs, nullptr, vram_out, preprocess_vram);
}

void HdSprites::frame_impl(const Machine& m, uint32_t* refs, const uint8_t* vram_snapshot, uint8_t* vram_out,
                           const std::function<void(uint8_t*)>& preprocess_vram) {
    frames_++;
    const int64_t now = now_ms();
    for (int c = 0; c < 256; c++)
        if (std::memcmp(last_dac_[c], m.dac[c], 3) != 0) {
            std::memcpy(last_dac_[c], m.dac[c], 3);
            colour_changed_[c] = now;
        }

    // Snapshot unsettled pictures under the lock, then perform hashing and filesystem existence checks outside
    // mutex_ so the game thread never stalls on draw() or present_buffer() during screen fades.
    struct ReadyImage {
        Deferred d;
        uint64_t matched_key = 0;
        bool has_matched_dac = false;
        std::array<std::array<uint8_t, 3>, 256> matched_dac{};
    };
    std::vector<Deferred> pending_images;
    std::filesystem::path folder_copy;
    bool has_disk_images = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        folder_copy = folder_;
        has_disk_images = !disk_images_.empty();
        for (const auto& d : deferred_)
            if (!d.settling) pending_images.push_back(d);
    }
    std::vector<ReadyImage> ready;
    if (!pending_images.empty()) {
        const auto cur_dac = copy_dac(m);
        for (const auto& d : pending_images) {
            std::bitset<256> colours;
            bool visible = false;
            for (uint8_t c : d.pixels) {
                colours.set(c);
                visible = visible || (cur_dac[c][0] | cur_dac[c][1] | cur_dac[c][2]);
            }
            if (!visible) continue;
            const uint64_t cur_id = image_key_of(d.pixels, d.width, cur_dac);
            bool instant_upscaled = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto kit = by_key_.find(cur_id);
                if (kit != by_key_.end()) {
                    const State st = slots_[kit->second].state;
                    instant_upscaled = st == State::unloaded || st == State::pending || st == State::placed;
                }
            }
            if (!instant_upscaled && !folder_copy.empty()) {
                wchar_t name[32];
                swprintf(name, 32, L"%016llx.png", static_cast<unsigned long long>(cur_id));
                std::error_code ec;
                instant_upscaled = std::filesystem::exists(folder_copy / L"upscaled" / name, ec);
            }
            if (instant_upscaled) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = std::find_if(deferred_.begin(), deferred_.end(), [&](const Deferred& item) { return item.seq == d.seq && !item.settling; });
                if (it != deferred_.end()) {
                    it->settling = true;
                    ready.push_back({*it, cur_id, false, {}});
                }
                continue;
            }
            if (has_disk_images) {
                std::vector<std::array<uint8_t, 3>> cur_pal;
                std::array<int16_t, 256> dac_to_canon;
                const uint64_t shape = canonical_image_shape_indexed(d.pixels, d.width, d.rows, cur_dac, &cur_pal, &dac_to_canon);
                if (shape && !cur_pal.empty()) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto sit = disk_images_.find(shape);
                    if (sit != disk_images_.end()) {
                        const DiskShape* best = nullptr;
                        float best_err = 1e9f;
                        for (const auto& cand : sit->second) {
                            if (cand.width != d.width || cand.rows != d.rows) continue;
                            float err = 0.0f;
                            if (match_near_exact(cand.colours, cur_pal)) {
                                best = &cand;
                                best_err = 0.0f;
                                break;
                            }
                            float tint[3];
                            if (match_ambient_dim(cand.colours, cur_pal, tint, &err) && err < best_err) {
                                best = &cand;
                                best_err = err;
                            }
                        }
                        if (best) {
                            auto it = std::find_if(deferred_.begin(), deferred_.end(),
                                                   [&](const Deferred& item) { return item.seq == d.seq && !item.settling; });
                            if (it != deferred_.end()) {
                                auto day_dac = cur_dac;
                                for (int c = 0; c < 256; c++) {
                                    const int16_t ci = dac_to_canon[size_t(c)];
                                    if (ci >= 0 && size_t(ci) < best->colours.size()) day_dac[size_t(c)] = best->colours[size_t(ci)];
                                }
                                it->settling = true;
                                ready.push_back({*it, best->key, true, day_dac});
                                continue;
                            }
                        }
                    }
                }
            }
            // A new picture without an upscaled replacement waits until it is faded in and settled.
            if (settled(colours)) {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = std::find_if(deferred_.begin(), deferred_.end(), [&](const Deferred& item) { return item.seq == d.seq && !item.settling; });
                if (it != deferred_.end()) {
                    it->settling = true;
                    ready.push_back({*it, 0, false, {}});
                }
            }
        }
    }
    for (const auto& r : ready) take_over(m, r.d, r.matched_key, r.has_matched_dac ? &r.matched_dac : nullptr);

    std::lock_guard<std::mutex> lock(mutex_);
    // First pass: check exact palette matches and detect any uniform ambient dimming (nighttime / weather)
    // from multi-channel placed pictures (preferring the largest backdrop).
    float global_tint[3] = {1.0f, 1.0f, 1.0f};
    bool has_global_dim = false;
    int best_dim_area = 0;
    for (auto& s : slots_) {
        if (s.state != State::placed) continue;
        if (palette_of(m, s.colours) == s.palette || match_slot_near_exact(m, s.colours, s.ref_dac)) {
            s.current = true;
            s.dimmed = false;
            s.tint[0] = s.tint[1] = s.tint[2] = 1.0f;
            continue;
        }
        float tint[3];
        if (match_slot_ambient(m, s.colours, s.ref_dac, tint)) {
            s.current = true;
            s.dimmed = true;
            s.tint[0] = tint[0];
            s.tint[1] = tint[1];
            s.tint[2] = tint[2];
            const int area = s.width * s.rows;
            if (area > best_dim_area) {
                best_dim_area = area;
                global_tint[0] = tint[0];
                global_tint[1] = tint[1];
                global_tint[2] = tint[2];
                has_global_dim = true;
            }
        } else {
            s.current = false;
            s.dimmed = false;
        }
    }
    // Second pass: single-channel sprites (e.g. monochrome red lava) can still match if they fit the active
    // screen-wide ambient dimming tint detected above.
    if (has_global_dim) {
        for (auto& s : slots_) {
            if (s.state != State::placed || s.current) continue;
            float tint[3];
            if (match_slot_ambient(m, s.colours, s.ref_dac, tint, global_tint)) {
                s.current = true;
                s.dimmed = true;
                s.tint[0] = tint[0];
                s.tint[1] = tint[1];
                s.tint[2] = tint[2];
            }
        }
    }
    const uint32_t frame_no = ++frame_count_;
    uint64_t cells = 0;
    auto ref_of = [&](const Mark& mark, uint8_t colour) -> uint32_t {
        if (!mark.slot || mark.slot > slots_.size() || colour != mark.colour) return kNone;
        Slot& s = slots_[mark.slot - 1];
        // A picture that shows on screen but is not in the atlas is asked for.
        if (s.state == State::unloaded) s.state = State::pending;
        if (s.seen != frame_no && (s.state == State::pending || s.state == State::placed)) {
            s.seen = frame_no;
            cells += uint64_t(s.width + 2) * uint64_t(s.rows + 2);
        }
        if (s.state != State::placed || !s.current) return kNone;
        return encode((s.atlas & 0xFFFF) + mark.x, (s.atlas >> 16) + mark.y, mark.order, s.dimmed);
    };
    const uint32_t source = screen_source_.load(std::memory_order_acquire);
    const bool check_vram = vram_dirty_.load(std::memory_order_acquire) || source == kVideo;
    const bool use_front = has_front_ && source != 0 && source != kVideo && source == front_source_;
    const uint8_t* vram = vram_snapshot;
    if (!vram) {
        const uint8_t* src_vram = presenting_ ? stable_vram_.data() : &m.mem[kVideo];
        if (vram_out) {
            std::memcpy(vram_out, src_vram, size_t(kPixels));
            if (preprocess_vram) preprocess_vram(vram_out);
            vram = vram_out;
        } else {
            vram = src_vram;
        }
    }
    for (int i = 0; i < kPixels; i++) {
        refs[i] = kNone;
        const uint8_t colour = vram[uint32_t(i)];
        // Sprites or pictures drawn directly to video memory (such as menus, sub-screens, and active buttons)
        // take precedence over the offscreen buffer until the buffer is copied back to video memory.
        if (check_vram) {
            const uint32_t vpos = kVideo + uint32_t(i);
            refs[i] = ref_of(sprite_marks_[vpos], colour);
            if (refs[i] == kNone) refs[i] = ref_of(image_marks_[vpos], colour);
            if (refs[i] != kNone || vram_masked_[size_t(i)] || source == kVideo) continue;
        }
        // When present_buffer() has snapshotted the offscreen buffer, read from the stable front marks so
        // concurrent offscreen redraws for the next frame cannot mismatch the presented VRAM pixels.
        if (use_front) {
            refs[i] = ref_of(front_sprite_marks_[size_t(i)], colour);
            if (refs[i] == kNone) refs[i] = ref_of(front_image_marks_[size_t(i)], colour);
            continue;
        }
        // Where the screen pixel was drawn: in the game's own picture buffer when it copies that to the screen
        // as a whole, otherwise wherever its value came from.
        uint32_t at;
        if (source) {
            at = source + uint32_t(i);
        } else {
            const uint32_t from = m.origin[kVideo + uint32_t(i)];
            if (!from) continue;
            at = from - 1;
        }
        at &= 0xFFFFF;
        // A sprite drawn over a picture wins while it is still there; when the game restores the picture
        // underneath, the colours match the picture again.
        refs[i] = ref_of(sprite_marks_[at], colour);
        if (refs[i] == kNone) refs[i] = ref_of(image_marks_[at], colour);
    }
    on_screen_cells_ = cells;
    // Pick the ambient lighting tint for this frame from the largest dimmed HD picture actually visible on
    // screen right now; bright daytime screens (such as option.vgg) automatically return (1, 1, 1).
    float frame_tint[3] = {1.0f, 1.0f, 1.0f};
    int seen_dim_area = 0;
    for (const auto& s : slots_) {
        if (s.seen != frame_no || s.state != State::placed || !s.current || !s.dimmed) continue;
        const int area = s.width * s.rows;
        if (area > seen_dim_area) {
            seen_dim_area = area;
            frame_tint[0] = s.tint[0];
            frame_tint[1] = s.tint[1];
            frame_tint[2] = s.tint[2];
        }
    }
    ambient_tint_[0] = frame_tint[0];
    ambient_tint_[1] = frame_tint[1];
    ambient_tint_[2] = frame_tint[2];
}

void HdSprites::ambient_tint(float out[3]) {
    std::lock_guard<std::mutex> lock(mutex_);
    out[0] = ambient_tint_[0];
    out[1] = ambient_tint_[1];
    out[2] = ambient_tint_[2];
}

bool HdSprites::is_showing_offscreen_map(int* popup_x0, int* popup_y0, int* popup_x1, int* popup_y1, bool* popup_has_holes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (popup_x0) *popup_x0 = 0;
    if (popup_y0) *popup_y0 = 0;
    if (popup_x1) *popup_x1 = 0;
    if (popup_y1) *popup_y1 = 0;
    if (popup_has_holes) *popup_has_holes = false;
    const uint32_t src = screen_source_.load(std::memory_order_acquire);
    if (!has_front_ || src == 0 || src == kVideo || src != front_source_) return false;
    int px0 = 320, py0 = 200, px1 = 0, py1 = 0;
    bool has_popup = false;
    auto grow = [&](int x, int y) {
        has_popup = true;
        px0 = std::min(px0, x);
        py0 = std::min(py0, y);
        px1 = std::max(px1, x + 2);
        py1 = std::max(py1, y + 2);
    };
    const bool vd = vram_dirty_.load(std::memory_order_acquire);
    // Scan the map area above the bottom message bar (y = 14..167) first so bottom text clears never create a fake popup.
    if (vd) {
        for (int y = 14; y < 168; y += 2)
            for (int x = 0; x < 256; x += 2) {
                const uint32_t off = uint32_t(y) * 320 + uint32_t(x);
                const bool clear_popup = vram_masked_[off] == 2;
                // Ignore sync_image_rect overlays (vram_masked_ == 1) and compass/scroll buttons (x >= 214).
                const bool direct_sprite = x < 214 && vram_masked_[off] != 1 &&
                                           ((image_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0 ||
                                            (sprite_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0);
                if (clear_popup || direct_sprite) grow(x, y);
            }
    }
    for (int y = 14; y < 168; y += 2)
        for (int x = 0; x < 256; x += 2) {
            const uint32_t off = uint32_t(y) * 320 + uint32_t(x);
            if ((front_image_marks_[off].slot & 0x80000000u) != 0 || (front_sprite_marks_[off].slot & 0x80000000u) != 0)
                grow(x, y);
        }
    // Extend upward (y = 0..13), downward (y = 168..199), and rightward into top-right advisor portraits (x = 256..319, y = 0..79)
    // only when a popup already started inside the map area above y = 168.
    if (has_popup) {
        const int upper_x0 = px0, upper_x1 = px1;
        if (vd) {
            for (int y = 0; y < 14; y += 2)
                for (int x = upper_x0; x < upper_x1; x += 2)
                    if (vram_masked_[uint32_t(y) * 320 + uint32_t(x)] == 2) grow(x, y);
            for (int y = 168; y < 200; y += 2)
                for (int x = upper_x0; x < upper_x1; x += 2)
                    if (vram_masked_[uint32_t(y) * 320 + uint32_t(x)] == 2) grow(x, y);
            // Advisor / Intelligence Report screens place a framed portrait box at x = 216..319, y = 0..79 alongside the report.
            if (py0 < 14) {
                for (int y = 0; y < 80; y += 2)
                    for (int x = 216; x < 320; x += 2)
                        if (vram_masked_[uint32_t(y) * 320 + uint32_t(x)] == 2) grow(x, y);
            }
        }
        for (int y = 168; y < 200; y += 2)
            for (int x = upper_x0; x < upper_x1; x += 2) {
                const uint32_t off = uint32_t(y) * 320 + uint32_t(x);
                if ((front_image_marks_[off].slot & 0x80000000u) != 0 || (front_sprite_marks_[off].slot & 0x80000000u) != 0)
                    grow(x, y);
            }
        if (!popup_x0 || !popup_y0 || !popup_x1 || !popup_y1) return false;
        // Full-screen 320x200 overlay covering both top-right AND bottom-right HUD areas stays in classic 4:3 mode.
        // Advisor/Intelligence screens cover top-right (portrait) but leave bottom-right (y = 110..160, x = 288) uncovered.
        if (px1 - px0 >= 254 && py1 - py0 >= 176) {
            bool covers_top_right = false, covers_bot_right = false;
            for (int y = 0; y < 14 && !covers_top_right; y += 2) {
                const uint32_t off = uint32_t(y) * 320 + 288;
                if ((vd && vram_masked_[off] == 2) || (front_image_marks_[off].slot & 0x80000000u) != 0)
                    covers_top_right = true;
            }
            for (int y = 110; y < 160 && !covers_bot_right; y += 2) {
                const uint32_t off = uint32_t(y) * 320 + 288;
                if ((vd && vram_masked_[off] == 2) || (front_image_marks_[off].slot & 0x80000000u) != 0)
                    covers_bot_right = true;
            }
            if (covers_top_right && covers_bot_right) return false;
        }
        // Detect L-shaped popups (such as Intelligence Reports with an unmasked gap beneath the top-right portrait).
        if (popup_has_holes) {
            int unmasked = 0, sampled = 0;
            for (int y = py0 + 4; y < py1 - 4; y += 4)
                for (int x = px0 + 4; x < px1 - 4; x += 4) {
                    const uint32_t off = uint32_t(y) * 320 + uint32_t(x);
                    const bool direct_sprite = vd && vram_masked_[off] != 1 && (x < 214 || y < 80) &&
                                               ((image_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0 ||
                                                (sprite_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0);
                    const bool covered = (vd && vram_masked_[off] == 2) || direct_sprite ||
                                         (front_image_marks_[off].slot & 0x80000000u) != 0 ||
                                         (front_sprite_marks_[off].slot & 0x80000000u) != 0;
                    sampled++;
                    if (!covered) unmasked++;
                }
            *popup_has_holes = sampled > 0 && unmasked * 10 >= sampled;
        }
        *popup_x0 = px0;
        *popup_y0 = py0;
        *popup_x1 = px1;
        *popup_y1 = py1;
    }
    return true;
}

void HdSprites::mark_popup_holes(uint32_t* refs, int x0, int y0, int x1, int y1) {
    if (!refs || x1 <= x0 || y1 <= y0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    const bool vd = vram_dirty_.load(std::memory_order_acquire);
    const int cx0 = std::clamp(x0, 0, 320), cy0 = std::clamp(y0, 0, 200);
    const int cx1 = std::clamp(x1, 0, 320), cy1 = std::clamp(y1, 0, 200);
    for (int y = cy0; y < cy1; y++) {
        for (int x = cx0; x < cx1; x++) {
            // Preserve HD button refs inside the 4 lower visible rows of the right command sidebar (268..315, 79..158);
            // the popup shader discards that rect separately for hole-aware popups so buttons never duplicate or drop to DOS.
            if (x >= 268 && x < 316 && y >= 79 && y < 159) continue;
            const uint32_t off = uint32_t(y) * 320 + uint32_t(x);
            const bool direct_sprite = vd && vram_masked_[off] != 1 && (x < 214 || y < 80) &&
                                       ((image_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0 ||
                                        (sprite_marks_[kVideo + off].slot & 0x7FFFFFFFu) != 0);
            const bool covered = (vd && vram_masked_[off] == 2) || direct_sprite ||
                                 (front_image_marks_[off].slot & 0x80000000u) != 0 ||
                                 (front_sprite_marks_[off].slot & 0x80000000u) != 0;
            if (!covered) refs[off] = kPopupHole;
        }
    }
}

void HdSprites::begin_world_frame() {
    std::lock_guard<std::mutex> lock(mutex_);
    world_frame_count_++;
    world_on_screen_cells_ = 0;
}

HdSprites::WorldSpriteInfo HdSprites::resolve_world_sprite(const Machine& m, uint32_t data, int width, int rows, uint8_t base,
                                                           bool bottom_up) {
    WorldSpriteInfo info;
    if (width <= 0 || width > 256 || rows <= 0 || rows > 256) return info;
    const uint32_t stride = uint32_t(width / 2);
    std::bitset<256> used;
    uint64_t id = kFnvBasis;
    auto mix = [](uint64_t& h, uint8_t v) { h = (h ^ v) * kFnvPrime; };
    if (!bottom_up) mix(id, 0xFE);
    for (uint32_t i = 0; i < stride * uint32_t(rows); i++) {
        const uint8_t byte = m.mem[(data + i) & 0xFFFFF];
        mix(id, byte);
        if (byte >> 4) used.set(uint8_t(base + (byte >> 4)));
        if (byte & 0x0F) used.set(uint8_t(base + (byte & 0x0F)));
    }
    mix(id, uint8_t(width));
    for (int c = 1; c < 16; c++)
        if (used[uint8_t(base + c)])
            for (int k = 0; k < 3; k++) mix(id, m.dac[uint8_t(base + c)][k]);
    info.key = id;

    auto check_slot = [&](Slot& s, uint32_t world_frame_no) -> bool {
        if (s.state == State::unloaded) s.state = State::pending;
        if (s.world_seen != world_frame_no && (s.state == State::pending || s.state == State::placed)) {
            s.world_seen = world_frame_no;
            // Avoid double-counting slots already accounted for in on_screen_cells_ during the current 2D frame.
            if (frame_count_ == 0 || s.seen != frame_count_) world_on_screen_cells_ += uint64_t(s.width + 2) * uint64_t(s.rows + 2);
        }
        if (s.state == State::placed) {
            // Re-validate palette if needed (slots added offscreen may not have been checked yet).
            if (palette_of(m, s.colours) == s.palette || match_slot_near_exact(m, s.colours, s.ref_dac)) {
                s.current = true;
                s.dimmed = false;
            } else {
                float tint[3];
                const bool have_hint = ambient_tint_[0] < 0.96f || ambient_tint_[1] < 0.96f || ambient_tint_[2] < 0.96f;
                if (match_slot_ambient(m, s.colours, s.ref_dac, tint, have_hint ? ambient_tint_ : nullptr)) {
                    s.current = true;
                    s.dimmed = true;
                } else {
                    s.current = false;
                }
            }
            if (s.current) {
                info.placed = true;
                info.dimmed = s.dimmed;
                info.atlas_x = uint16_t(s.atlas & 0xFFFF);
                info.atlas_y = uint16_t(s.atlas >> 16);
                return true;
            }
        }
        return s.state == State::pending || s.state == State::failed;
    };

    std::filesystem::path exact_up;
    float hint[3] = {1.0f, 1.0f, 1.0f};
    bool have_hint = false;
    bool has_candidates = false;
    uint64_t shapes_ver = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shapes_ver = disk_shapes_version_;
        if (ambient_tint_[0] < 0.96f || ambient_tint_[1] < 0.96f || ambient_tint_[2] < 0.96f) {
            hint[0] = ambient_tint_[0];
            hint[1] = ambient_tint_[1];
            hint[2] = ambient_tint_[2];
            have_hint = true;
        }
        auto it = by_key_.find(id);
        if (it != by_key_.end()) {
            Slot& s = slots_[it->second];
            if (check_slot(s, world_frame_count_)) return info;
            if (s.state == State::missing && s.miss_shape_version == shapes_ver) return info;
        }
        auto wit = world_resolve_cache_.find(id);
        if (wit != world_resolve_cache_.end()) {
            const WorldResolveCache& wc = wit->second;
            if (wc.slot != UINT32_MAX && wc.slot < slots_.size()) {
                if (check_slot(slots_[wc.slot], world_frame_count_)) return info;
                // Palette drifted away from cached alias; re-evaluate below.
                world_resolve_cache_.erase(wit);
            } else if (wc.slot == UINT32_MAX && wc.shapes_version == shapes_ver &&
                       std::abs(wc.hint[0] - hint[0]) < 0.04f &&
                       std::abs(wc.hint[1] - hint[1]) < 0.04f &&
                       std::abs(wc.hint[2] - hint[2]) < 0.04f) {
                return info;
            }
        }
        has_candidates = !disk_shapes_.empty();
        if (!folder_.empty()) exact_up = upscaled_file(id);
    }

    auto cache_world_result = [&](uint32_t slot_idx) {
        if (world_resolve_cache_.size() >= 4096) world_resolve_cache_.clear();
        world_resolve_cache_[id] = {slot_idx, shapes_ver, {hint[0], hint[1], hint[2]}};
    };

    // Check if an exact upscaled PNG exists on disk (never saving new original/ PNGs from offscreen tiles).
    std::error_code ec;
    if (!exact_up.empty() && std::filesystem::exists(exact_up, ec)) {
        const auto ref_dac = copy_dac(m);
        const uint64_t pal_hash = palette_of(m, used);
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = by_key_.find(id);
        uint32_t slot = 0;
        if (it != by_key_.end()) {
            slot = it->second;
            if (slots_[slot].state == State::missing) slots_[slot].state = State::pending;
        } else {
            slot = uint32_t(slots_.size());
            slots_.push_back({id, id, width, rows, State::pending, 0, used, pal_hash, ref_dac, true});
            by_key_.emplace(id, slot);
        }
        cache_world_result(slot);
        check_slot(slots_[slot], world_frame_count_);
        return info;
    }

    // Fall back to canonical shape matching against already-upscaled daytime sprites on disk (+-2 drift or night dim).
    if (!has_candidates) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_world_result(UINT32_MAX);
        return info;
    }
    std::vector<uint32_t> rgba(size_t(width) * size_t(rows));
    for (int y = 0; y < rows; y++) {
        const uint32_t row = data + uint32_t(bottom_up ? rows - 1 - y : y) * stride;
        for (int x = 0; x < width; x++) {
            const uint8_t c = nibble(m, row, x);
            rgba[size_t(y) * size_t(width) + size_t(x)] = c ? colour_of(m, uint8_t(base + c)) : 0;
        }
    }
    std::vector<std::array<uint8_t, 3>> computed_pal;
    const uint64_t shape = canonical_shape(rgba, width, rows, &computed_pal);
    if (!shape || computed_pal.empty()) {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_world_result(UINT32_MAX);
        return info;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto sit = disk_shapes_.find(shape);
    if (sit == disk_shapes_.end()) {
        cache_world_result(UINT32_MAX);
        return info;
    }
    const DiskShape* best = nullptr;
    float best_err = 1e9f;
    auto scan = [&](bool near_exact_only, const float* use_hint) {
        for (const auto& cand : sit->second) {
            if (!cand.has_upscaled || cand.width != width || cand.rows != rows) continue;
            float err = 0.0f;
            if (near_exact_only) {
                if (!match_near_exact(cand.colours, computed_pal)) continue;
            } else {
                float tint[3];
                if (!match_ambient_dim(cand.colours, computed_pal, tint, &err, use_hint)) continue;
            }
            if (!best || err < best_err) {
                best = &cand;
                best_err = err;
            }
        }
    };
    scan(true, nullptr);
    if (!best) scan(false, nullptr);
    if (!best && have_hint) scan(false, hint);
    if (!best) {
        cache_world_result(UINT32_MAX);
        return info;
    }

    auto kit = by_key_.find(best->key);
    if (kit != by_key_.end()) {
        cache_world_result(kit->second);
        check_slot(slots_[kit->second], world_frame_count_);
        return info;
    }
    auto ref_dac = copy_dac(m);
    std::vector<uint32_t> seen_rgb;
    for (int y = 0; y < rows; y++) {
        const uint32_t row = data + uint32_t(bottom_up ? rows - 1 - y : y) * stride;
        for (int x = 0; x < width; x++) {
            const uint8_t c = nibble(m, row, x);
            if (!c) continue;
            const uint32_t rgb = colour_of(m, uint8_t(base + c)) & 0x00FFFFFFu;
            size_t ci = 0;
            while (ci < seen_rgb.size() && seen_rgb[ci] != rgb) ci++;
            if (ci == seen_rgb.size()) seen_rgb.push_back(rgb);
            if (ci < best->colours.size()) ref_dac[size_t(uint8_t(base + c))] = best->colours[ci];
        }
    }
    const uint64_t pal_hash = palette_of_dac(ref_dac, used);
    const uint32_t slot = uint32_t(slots_.size());
    slots_.push_back({best->key, best->key, width, rows, State::pending, 0, used, pal_hash, ref_dac, true});
    by_key_.emplace(best->key, slot);
    cache_world_result(slot);
    check_slot(slots_[slot], world_frame_count_);
    return info;
}

uint64_t HdSprites::on_screen_cells() {
    std::lock_guard<std::mutex> lock(mutex_);
    return on_screen_cells_ + world_on_screen_cells_;
}

bool AtlasClearing::overflow(int64_t now_ms, uint64_t on_screen) {
    // About as much: between three quarters and five quarters.
    auto about = [](uint64_t need, uint64_t than) { return need * 4 >= than * 3 && need * 4 <= than * 5; };
    // Each hold runs out a minute after it was set. Overflows in between, which come every frame while a
    // picture waits for room, do not extend it.
    std::erase_if(held_, [now_ms](const Held& h) { return now_ms >= h.until; });
    if (now_ms - cleared_at_ < 10000) {
        // Full again soon after emptying. If the screen still needs about as much as then, emptying did not help.
        if (about(on_screen, cleared_for_) &&
            std::none_of(held_.begin(), held_.end(), [&](const Held& h) { return h.need == cleared_for_; }))
            held_.push_back({cleared_for_, now_ms + 60000});
        return false;
    }
    if (on_screen > capacity_ / 2) return false;
    if (std::any_of(held_.begin(), held_.end(), [&](const Held& h) { return about(on_screen, h.need); })) return false;
    cleared_at_ = now_ms;
    cleared_for_ = on_screen;
    return true;
}

std::vector<HdSprites::Pending> HdSprites::take_pending() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Pending> out;
    if (folder_.empty()) return out;
    // Pictures made while the game runs are picked up a moment later.
    const auto now = std::chrono::steady_clock::now();
    const bool scan = now >= next_scan_;
    if (scan) next_scan_ = now + std::chrono::seconds(2);
    for (uint32_t i = 0; i < slots_.size(); i++) {
        Slot& s = slots_[i];
        if (s.state == State::missing && scan) {
            std::error_code ec;
            // A picture written under the sprite's earlier name, by a redraw that started before the game
            // renamed the sprite, is moved to the new name.
            if (s.legacy != s.key) moved(upscaled_file(s.legacy), upscaled_file(s.key));
            if (std::filesystem::exists(upscaled_file(s.key), ec)) {
                s.state = State::unloaded;
                if (s.shape) {
                    auto sit = disk_shapes_.find(s.shape);
                    if (sit != disk_shapes_.end())
                        for (auto& d : sit->second)
                            if (d.key == s.key) d.has_upscaled = true;
                } else {
                    for (auto& [sh, list] : disk_shapes_)
                        for (auto& d : list)
                            if (d.key == s.key) {
                                d.has_upscaled = true;
                                s.shape = sh;
                            }
                }
                disk_shapes_version_++;
            }
        }
        if (s.state == State::pending) out.push_back({i, upscaled_file(s.key), s.width, s.rows});
    }
    return out;
}

void HdSprites::placed(uint32_t slot, uint16_t x, uint16_t y) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot >= slots_.size()) return;
    slots_[slot].state = State::placed;
    slots_[slot].atlas = uint32_t(x) | (uint32_t(y) << 16);
}

void HdSprites::failed(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < slots_.size()) slots_[slot].state = State::failed;
}

void HdSprites::no_room(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot < slots_.size() && slots_[slot].state == State::pending) slots_[slot].state = State::unloaded;
}

void HdSprites::atlas_cleared() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& s : slots_)
        if (s.state == State::placed || s.state == State::pending) s.state = State::unloaded;
}

size_t HdSprites::sprite_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return slots_.size();
}

size_t HdSprites::missing_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (const auto& s : slots_) n += s.state == State::missing ? 1 : 0;
    return n;
}

}  // namespace nax
