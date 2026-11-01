// DOS services (INT 21h): files, memory, vectors, time and program exit.
//
// Original game files are never changed. Reads look in the save folder first and then in the game folder.
// Every write goes to a temporary file in the save folder, which replaces the target only when the program
// closes the file, so an interrupted write never leaves a half-written file behind.
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <ctime>
#include <io.h>

#include "machine.h"

namespace nax {

namespace {

std::string lower(std::string s) {
    for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

bool wildcard_match(const std::string& pattern, const std::string& name) {
    // DOS 8.3 matching: "*" ends the name or extension part, "?" matches one character.
    auto split = [](const std::string& s) {
        const size_t dot = s.find('.');
        return std::make_pair(s.substr(0, dot), dot == std::string::npos ? std::string() : s.substr(dot + 1));
    };
    auto part = [](const std::string& p, const std::string& n) {
        size_t i = 0;
        for (; i < p.size(); i++) {
            if (p[i] == '*') return true;
            if (i >= n.size()) return false;
            if (p[i] != '?' && std::tolower(static_cast<unsigned char>(p[i])) != std::tolower(static_cast<unsigned char>(n[i]))) return false;
        }
        return i == n.size();
    };
    const auto [pn, pe] = split(pattern);
    const auto [nn, ne] = split(name);
    return part(pn, nn) && part(pe, ne);
}

// Rejects path parts that Windows would read as a drive, an alternate data stream or a device.
bool safe_part(const std::string& part) {
    for (unsigned char ch : part)
        if (ch < 32 || ch >= 127 || std::strchr("<>:\"|?*", ch)) return false;
    const std::string base = lower(part.substr(0, part.find('.')));
    static const char* devices[] = {"con", "prn", "aux", "nul", "clock$"};
    for (const char* d : devices)
        if (base == d) return false;
    if (base.size() == 4 && (base.rfind("com", 0) == 0 || base.rfind("lpt", 0) == 0) && base[3] >= '0' && base[3] <= '9') return false;
    return true;
}

}  // namespace

std::string Machine::read_string(uint16_t s, uint16_t o, size_t max) const {
    std::string out;
    for (size_t i = 0; i < max; i++) {
        const char ch = char(rd8(s, uint16_t(o + i)));
        if (!ch) break;
        out += ch;
    }
    return out;
}

bool Machine::resolve(const std::string& dos_name, bool for_write, std::filesystem::path* out) {
    std::string name = dos_name;
    if (name.size() >= 2 && name[1] == ':') name = name.substr(2);
    std::replace(name.begin(), name.end(), '\\', '/');
    std::string full = (!name.empty() && name[0] == '/') ? name.substr(1) : (cwd_.empty() ? name : cwd_ + "/" + name);
    std::filesystem::path rel;
    size_t start = 0;
    while (start <= full.size()) {
        const size_t end = full.find('/', start);
        std::string part = full.substr(start, end == std::string::npos ? std::string::npos : end - start);
        // DOS ignores trailing spaces and dots, and so does Windows, so "CON " has to be checked as "CON".
        // Utopia itself asks for "option.vgg " with a trailing space.
        if (part != "." && part != "..")
            while (!part.empty() && (part.back() == ' ' || part.back() == '.')) part.pop_back();
        if (part == ".." || !safe_part(part)) return false;  // nothing may reach outside the game folder
        if (!part.empty() && part != ".") rel /= std::filesystem::path(lower(part));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (rel.empty()) return false;
    const auto overlay = save_dir_ / L"files" / rel;
    std::error_code ec;
    if (for_write) {
        std::filesystem::create_directories(overlay.parent_path(), ec);
        *out = overlay;
        return true;
    }
    if (std::filesystem::exists(overlay, ec)) { *out = overlay; return true; }
    const auto original = data_dir_ / rel;
    if (std::filesystem::exists(original, ec)) { *out = original; return true; }
    return false;
}

bool Machine::close_file(int handle, bool commit) {
    auto it = files_.find(handle);
    if (it == files_.end()) return false;
    File& file = it->second;
    bool written = true;
    if (file.f) {
        // A short write sets the stream's error flag even when the final flush succeeds, so that is checked too.
        written = !std::ferror(file.f);
        written = std::fflush(file.f) == 0 && written;
        written = std::fclose(file.f) == 0 && written;
    }
    bool ok = true;
    if (!file.temp_path.empty()) {
        std::error_code ec;
        if (!commit || !written) {
            std::filesystem::remove(file.temp_path, ec);
            log("Discarded unfinished writes to %s; the previous version is unchanged", file.name.c_str());
            ok = !commit;
        } else if (!MoveFileExW(file.temp_path.c_str(), file.final_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            log("Could not save %s (error %lu); the previous version is unchanged", file.name.c_str(), GetLastError());
            std::filesystem::remove(file.temp_path, ec);
            ok = false;
        } else {
            log("Saved %s", file.name.c_str());
        }
    }
    files_.erase(it);
    return ok;
}

void Machine::finish_files(bool commit) {
    while (!files_.empty()) close_file(files_.begin()->first, commit);
}

uint16_t Machine::alloc(uint16_t paras, uint16_t* largest) {
    std::sort(blocks_.begin(), blocks_.end(), [](const Block& a, const Block& b) { return a.seg < b.seg; });
    uint32_t cursor = kFirstFreeSegment;
    uint32_t best = 0;
    for (size_t i = 0; i <= blocks_.size(); i++) {
        const uint32_t next = i < blocks_.size() ? blocks_[i].seg : kMemoryTop;
        if (next >= cursor) {
            const uint32_t gap = next - cursor;
            if (gap >= paras && paras > 0) {
                blocks_.push_back({uint16_t(cursor), paras});
                return uint16_t(cursor);
            }
            best = std::max(best, gap);
        }
        if (i < blocks_.size()) cursor = std::max<uint32_t>(cursor, uint32_t(blocks_[i].seg) + blocks_[i].paras);
    }
    *largest = uint16_t(best);
    return 0;
}

bool Machine::resize(uint16_t seg, uint16_t paras, uint16_t* largest) {
    std::sort(blocks_.begin(), blocks_.end(), [](const Block& a, const Block& b) { return a.seg < b.seg; });
    for (size_t i = 0; i < blocks_.size(); i++) {
        if (blocks_[i].seg != seg) continue;
        const uint32_t limit = i + 1 < blocks_.size() ? blocks_[i + 1].seg : kMemoryTop;
        if (uint32_t(seg) + paras > limit) {
            *largest = uint16_t(limit - seg);
            return false;
        }
        blocks_[i].paras = paras;
        return true;
    }
    *largest = 0;
    return false;
}

bool Machine::free_block(uint16_t seg) {
    for (size_t i = 0; i < blocks_.size(); i++) {
        if (blocks_[i].seg == seg) {
            blocks_.erase(blocks_.begin() + long(i));
            return true;
        }
    }
    return false;
}

void Machine::find_entry(const std::string& name, uintmax_t size) {
    // Fills the DOS disk transfer area the way function 4Eh does.
    const uint16_t s = dta_seg_, o = dta_off_;
    wr8(s, uint16_t(o + 0x15), 0x20);
    wr16(s, uint16_t(o + 0x16), 0);
    wr16(s, uint16_t(o + 0x18), 0x21);
    wr16(s, uint16_t(o + 0x1A), uint16_t(size));
    wr16(s, uint16_t(o + 0x1C), uint16_t(size >> 16));
    std::string upper = name;
    for (char& ch : upper) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    for (size_t i = 0; i < 13; i++) wr8(s, uint16_t(o + 0x1E + i), i < upper.size() ? uint8_t(upper[i]) : 0);
}

void Machine::int21() {
    c.cf = false;
    switch (c.a.h) {
    case 0x02:
        log("DOS output: %c", c.d.l);
        return;
    case 0x09: {
        std::string text;
        for (uint16_t i = 0; i < 1024; i++) {
            const char ch = char(rd8(c.ds, uint16_t(c.d.x + i)));
            if (ch == '$') break;
            text += ch;
        }
        log("DOS output: %s", text.c_str());
        return;
    }
    case 0x0B: c.a.l = bios_keys_.empty() ? 0 : 0xFF; return;
    case 0x0E: c.a.l = 3; return;
    case 0x19: c.a.l = 2; return;
    case 0x1A: dta_seg_ = c.ds; dta_off_ = c.d.x; return;
    case 0x25:
        wr16(0, uint16_t(c.a.l * 4), c.d.x);
        wr16(0, uint16_t(c.a.l * 4 + 2), c.ds);
        log("Interrupt %02x now points to %04x:%04x", c.a.l, c.ds, c.d.x);
        return;
    case 0x2A: {
        std::time_t t = std::time(nullptr);
        std::tm lt{};
        localtime_s(&lt, &t);
        c.c.x = uint16_t(lt.tm_year + 1900);
        c.d.h = uint8_t(lt.tm_mon + 1);
        c.d.l = uint8_t(lt.tm_mday);
        c.a.l = uint8_t(lt.tm_wday);
        return;
    }
    case 0x2C: {
        SYSTEMTIME st;
        GetLocalTime(&st);
        c.c.h = uint8_t(st.wHour);
        c.c.l = uint8_t(st.wMinute);
        c.d.h = uint8_t(st.wSecond);
        c.d.l = uint8_t(st.wMilliseconds / 10);
        return;
    }
    case 0x2F: c.es = dta_seg_; c.b.x = dta_off_; return;
    case 0x30: c.a.x = 0x0005; c.b.x = 0; c.c.x = 0; return;
    case 0x33: c.d.l = 0; return;
    case 0x35: {
        const uint32_t v = far_vector(c.a.l);
        c.b.x = uint16_t(v);
        c.es = uint16_t(v >> 16);
        return;
    }
    case 0x36: c.a.x = 64; c.b.x = 0x7FFF; c.c.x = 512; c.d.x = 0xFFFF; return;
    case 0x3B: {
        std::string dir = read_string(c.ds, c.d.x);
        if (dir.size() >= 2 && dir[1] == ':') dir = dir.substr(2);
        std::replace(dir.begin(), dir.end(), '\\', '/');
        if (!dir.empty() && dir[0] == '/') cwd_.clear();
        std::string next = cwd_;
        size_t start = 0;
        while (start <= dir.size()) {
            const size_t end = dir.find('/', start);
            const std::string part = dir.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (part == "..") {
                const size_t slash = next.rfind('/');
                next = slash == std::string::npos ? std::string() : next.substr(0, slash);
            } else if (!part.empty() && part != ".") {
                std::string clean = part;
                while (!clean.empty() && (clean.back() == ' ' || clean.back() == '.')) clean.pop_back();
                if (!safe_part(clean)) { dos_error(3); return; }  // the same rule as for file names
                next = next.empty() ? lower(clean) : next + "/" + lower(clean);
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
        std::error_code ec;
        if (!next.empty() && !std::filesystem::is_directory(data_dir_ / std::filesystem::path(next), ec) &&
            !std::filesystem::is_directory(save_dir_ / L"files" / std::filesystem::path(next), ec)) {
            dos_error(3);
            return;
        }
        cwd_ = next;
        log("Changed directory to \\%s", cwd_.c_str());
        return;
    }
    case 0x3C: case 0x3D: {
        const std::string name = read_string(c.ds, c.d.x);
        const bool create = c.a.h == 0x3C;
        const int mode = create ? 2 : (c.a.l & 7);
        std::filesystem::path path;
        int handle = 5;
        while (files_.count(handle)) handle++;
        File file;
        file.name = name;
        if (mode == 0) {
            if (!resolve(name, false, &path)) { log("Open %s: not found", name.c_str()); dos_error(2); return; }
            file.f = _wfopen(path.c_str(), L"rb");
        } else {
            std::filesystem::path existing;
            const bool had = !create && resolve(name, false, &existing);
            if (!create && !had) { log("Open %s for writing: not found", name.c_str()); dos_error(2); return; }
            if (!resolve(name, true, &file.final_path)) { dos_error(3); return; }
            // Each handle writes its own temporary file, so two handles on one file never overwrite each other.
            file.temp_path = file.final_path;
            file.temp_path += L".nax-tmp" + std::to_wstring(handle);
            std::error_code ec;
            if (had && !std::filesystem::copy_file(existing, file.temp_path, std::filesystem::copy_options::overwrite_existing, ec)) {
                log("Open %s for writing: could not copy the current version", name.c_str());
                dos_error(5);
                return;
            }
            file.f = _wfopen(file.temp_path.c_str(), had ? L"r+b" : L"w+b");
        }
        if (!file.f) { log("Open %s: failed", name.c_str()); dos_error(5); return; }
        log("%s %s as handle %d", create ? "Created" : (mode == 0 ? "Opened" : "Opened for writing"), name.c_str(), handle);
        files_[handle] = std::move(file);
        c.a.x = uint16_t(handle);
        return;
    }
    case 0x3E:
        if (c.b.x < 5) return;
        if (!files_.count(c.b.x)) { dos_error(6); return; }
        if (!close_file(c.b.x, true)) dos_error(5);
        return;
    case 0x3F: {
        if (c.b.x == 0) { c.a.x = 0; return; }
        auto it = files_.find(c.b.x);
        if (it == files_.end()) { dos_error(6); return; }
        std::vector<uint8_t> buf(c.c.x);
        const size_t n = std::fread(buf.data(), 1, buf.size(), it->second.f);
        for (size_t i = 0; i < n; i++) wr8(c.ds, uint16_t(c.d.x + i), buf[i]);
        c.a.x = uint16_t(n);
        return;
    }
    case 0x40: {
        if (c.b.x == 1 || c.b.x == 2) {
            log("DOS output: %s", read_string(c.ds, c.d.x, c.c.x).c_str());
            c.a.x = c.c.x;
            return;
        }
        auto it = files_.find(c.b.x);
        if (it == files_.end()) { dos_error(6); return; }
        if (it->second.temp_path.empty()) { dos_error(5); return; }
        if (c.c.x == 0) {
            std::fflush(it->second.f);
            _chsize_s(_fileno(it->second.f), _ftelli64(it->second.f));
            c.a.x = 0;
            return;
        }
        std::vector<uint8_t> buf(c.c.x);
        for (size_t i = 0; i < buf.size(); i++) buf[i] = rd8(c.ds, uint16_t(c.d.x + i));
        c.a.x = uint16_t(std::fwrite(buf.data(), 1, buf.size(), it->second.f));
        return;
    }
    case 0x41: {
        const std::string name = read_string(c.ds, c.d.x);
        std::filesystem::path path;
        std::error_code ec;
        // Only copies in the save folder can be deleted. Original game files stay untouched.
        if (resolve(name, true, &path) && std::filesystem::remove(path, ec)) return;
        dos_error(5);
        return;
    }
    case 0x42: {
        auto it = files_.find(c.b.x);
        if (it == files_.end()) { dos_error(6); return; }
        const int64_t offset = int64_t(int32_t((uint32_t(c.c.x) << 16) | c.d.x));
        const int origin = c.a.l == 0 ? SEEK_SET : (c.a.l == 1 ? SEEK_CUR : SEEK_END);
        if (_fseeki64(it->second.f, offset, origin) != 0) { dos_error(25); return; }
        const int64_t pos = _ftelli64(it->second.f);
        c.a.x = uint16_t(pos);
        c.d.x = uint16_t(pos >> 16);
        return;
    }
    case 0x43: {
        std::filesystem::path path;
        if (!resolve(read_string(c.ds, c.d.x), false, &path)) { dos_error(2); return; }
        c.c.x = 0x20;
        return;
    }
    case 0x44:
        if (c.a.l == 0) { c.d.x = c.b.x < 5 ? 0x80D3 : 0x0002; return; }
        if (c.a.l == 8) { c.a.x = 1; return; }
        dos_error(1);
        return;
    case 0x47: {
        std::string dir = cwd_;
        std::replace(dir.begin(), dir.end(), '/', '\\');
        for (char& ch : dir) ch = char(std::toupper(static_cast<unsigned char>(ch)));
        for (size_t i = 0; i <= dir.size(); i++) wr8(c.ds, uint16_t(c.si + i), i < dir.size() ? uint8_t(dir[i]) : 0);
        return;
    }
    case 0x48: {
        uint16_t largest = 0;
        const uint16_t seg = alloc(c.b.x, &largest);
        if (!seg) { c.b.x = largest; dos_error(8); return; }
        c.a.x = seg;
        return;
    }
    case 0x49:
        if (!free_block(c.es)) dos_error(9);
        return;
    case 0x4A: {
        uint16_t largest = 0;
        if (!resize(c.es, c.b.x, &largest)) { c.b.x = largest; dos_error(8); }
        return;
    }
    case 0x4C:
        exit_code_ = c.a.l;
        running_ = false;
        log("Program exited with code %d", exit_code_);
        return;
    case 0x4E: case 0x4F: {
        if (c.a.h == 0x4E) {
            find_results_.clear();
            find_next_ = 0;
            std::string pattern = read_string(c.ds, c.d.x);
            if (pattern.size() >= 2 && pattern[1] == ':') pattern = pattern.substr(2);
            std::replace(pattern.begin(), pattern.end(), '\\', '/');
            const size_t slash = pattern.rfind('/');
            const std::string dir = slash == std::string::npos ? std::string() : pattern.substr(0, slash);
            const std::string mask = slash == std::string::npos ? pattern : pattern.substr(slash + 1);
            std::string base = dir.empty() ? cwd_ : (cwd_.empty() ? dir : cwd_ + "/" + dir);
            // A search may only look inside the game folder, under the same rule as file names.
            for (size_t from = 0; from <= base.size();) {
                const size_t to = base.find('/', from);
                const std::string part = base.substr(from, to == std::string::npos ? std::string::npos : to - from);
                if (part == ".." || (!part.empty() && part != "." && !safe_part(part))) { dos_error(3); return; }
                if (to == std::string::npos) break;
                from = to + 1;
            }
            for (const auto& root : {save_dir_ / L"files", data_dir_}) {
                std::error_code ec;
                const auto folder = base.empty() ? root : root / std::filesystem::path(lower(base));
                std::filesystem::directory_iterator it(folder, ec), end;
                for (; !ec && it != end; it.increment(ec)) {
                    const auto& entry = *it;
                    // Only plain ASCII 8.3 names are visible to the game; other names are skipped before conversion.
                    const std::wstring wide = entry.path().filename().wstring();
                    if (wide.size() > 12 || std::any_of(wide.begin(), wide.end(), [](wchar_t ch) { return ch < 32 || ch >= 127; })) continue;
                    std::string n;
                    for (wchar_t ch : wide) n += char(ch);
                    std::error_code type_ec;
                    if (!entry.is_regular_file(type_ec) || n.find(".nax-tmp") != std::string::npos) continue;
                    if (wildcard_match(mask, n) && std::find(find_results_.begin(), find_results_.end(), lower(n)) == find_results_.end())
                        find_results_.push_back(lower(n));
                }
            }
            std::sort(find_results_.begin(), find_results_.end());
            log("Search %s: %zu matches", pattern.c_str(), find_results_.size());
        }
        if (find_next_ >= find_results_.size()) { dos_error(18); return; }
        std::filesystem::path path;
        const std::string name = find_results_[find_next_++];
        std::error_code ec;
        const uintmax_t size = resolve(name, false, &path) ? std::filesystem::file_size(path, ec) : 0;
        find_entry(name, size);
        return;
    }
    default:
        if (!unknown_services_[0x2100 | c.a.h]) {
            unknown_services_[0x2100 | c.a.h] = true;
            log("Unhandled DOS function %02x (al=%02x)", c.a.h, c.a.l);
        }
        dos_error(1);
    }
}

}  // namespace nax
