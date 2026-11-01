// Flag-exact 8086 arithmetic used by translated code. Each helper returns the result and sets CF, ZF, SF and OF
// the way the original instruction does. Shift and rotate counts of zero leave the flags unchanged.
#pragma once

#include <type_traits>

#include "machine.h"

namespace nax {

template <int Bits>
struct Width {
    using T = std::conditional_t<Bits == 8, uint8_t, uint16_t>;
    static constexpr uint32_t mask = Bits == 8 ? 0xFFu : 0xFFFFu;
    static constexpr uint32_t sign = Bits == 8 ? 0x80u : 0x8000u;
};

template <int Bits>
inline typename Width<Bits>::T result_flags(Cpu& c, uint32_t r) {
    using W = Width<Bits>;
    const auto v = typename W::T(r & W::mask);
    c.zf = v == 0;
    c.sf = (v & W::sign) != 0;
    return v;
}

template <int Bits>
inline typename Width<Bits>::T addw(Cpu& c, uint32_t a, uint32_t b, bool carry) {
    using W = Width<Bits>;
    const uint32_t r = a + b + (carry ? 1 : 0);
    c.cf = r > W::mask;
    c.of = ((a ^ r) & (b ^ r) & W::sign) != 0;
    return result_flags<Bits>(c, r);
}

template <int Bits>
inline typename Width<Bits>::T subw(Cpu& c, uint32_t a, uint32_t b, bool borrow) {
    using W = Width<Bits>;
    const uint32_t r = a - b - (borrow ? 1 : 0);
    c.cf = a < b + (borrow ? 1 : 0);
    c.of = ((a ^ b) & (a ^ r) & W::sign) != 0;
    return result_flags<Bits>(c, r);
}

inline uint8_t add8(Cpu& c, uint8_t a, uint8_t b, bool carry) { return addw<8>(c, a, b, carry); }
inline uint16_t add16(Cpu& c, uint16_t a, uint16_t b, bool carry) { return addw<16>(c, a, b, carry); }
inline uint8_t sub8(Cpu& c, uint8_t a, uint8_t b, bool borrow) { return subw<8>(c, a, b, borrow); }
inline uint16_t sub16(Cpu& c, uint16_t a, uint16_t b, bool borrow) { return subw<16>(c, a, b, borrow); }

inline uint8_t logic8(Cpu& c, uint8_t r) { c.cf = c.of = false; return result_flags<8>(c, r); }
inline uint16_t logic16(Cpu& c, uint16_t r) { c.cf = c.of = false; return result_flags<16>(c, r); }

inline uint8_t inc8(Cpu& c, uint8_t a) { c.of = a == 0x7F; return result_flags<8>(c, a + 1u); }
inline uint16_t inc16(Cpu& c, uint16_t a) { c.of = a == 0x7FFF; return result_flags<16>(c, a + 1u); }
inline uint8_t dec8(Cpu& c, uint8_t a) { c.of = a == 0x80; return result_flags<8>(c, a - 1u); }
inline uint16_t dec16(Cpu& c, uint16_t a) { c.of = a == 0x8000; return result_flags<16>(c, a - 1u); }
inline uint8_t neg8(Cpu& c, uint8_t a) { return subw<8>(c, 0, a, false); }
inline uint16_t neg16(Cpu& c, uint16_t a) { return subw<16>(c, 0, a, false); }

template <int Bits>
inline typename Width<Bits>::T shlw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    const uint32_t r = v << n;
    c.cf = n <= unsigned(Bits) ? ((r >> Bits) & 1) != 0 : false;
    const auto out = result_flags<Bits>(c, r);
    c.of = ((out & W::sign) != 0) != c.cf;
    return out;
}

template <int Bits>
inline typename Width<Bits>::T shrw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    c.cf = n <= unsigned(Bits) ? ((v >> (n - 1)) & 1) != 0 : false;
    c.of = (v & W::sign) != 0;
    return result_flags<Bits>(c, n < 32 ? v >> n : 0);
}

template <int Bits>
inline typename Width<Bits>::T sarw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    const int32_t s = Bits == 8 ? int32_t(int8_t(v)) : int32_t(int16_t(v));
    const unsigned k = n < unsigned(Bits) ? n : unsigned(Bits);
    c.cf = ((s >> (k - 1)) & 1) != 0;
    c.of = false;
    return result_flags<Bits>(c, uint32_t(s >> k));
}

template <int Bits>
inline typename Width<Bits>::T rolw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    const unsigned k = n % Bits;
    const auto r = typename W::T(((v << k) | (v >> ((Bits - k) % Bits))) & W::mask);
    c.cf = (r & 1) != 0;
    c.of = ((r & W::sign) != 0) != c.cf;
    return r;
}

template <int Bits>
inline typename Width<Bits>::T rorw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    const unsigned k = n % Bits;
    const auto r = typename W::T(((v >> k) | (v << ((Bits - k) % Bits))) & W::mask);
    c.cf = (r & W::sign) != 0;
    c.of = ((r & W::sign) != 0) != ((r & (W::sign >> 1)) != 0);
    return r;
}

template <int Bits>
inline typename Width<Bits>::T rclw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    uint32_t r = v & W::mask;
    bool cf = c.cf;
    for (unsigned i = 0; i < n % (Bits + 1); i++) {
        const bool out = (r & W::sign) != 0;
        r = ((r << 1) | (cf ? 1 : 0)) & W::mask;
        cf = out;
    }
    c.cf = cf;
    c.of = ((r & W::sign) != 0) != cf;
    return typename W::T(r);
}

template <int Bits>
inline typename Width<Bits>::T rcrw(Cpu& c, uint32_t v, unsigned n) {
    using W = Width<Bits>;
    n &= 0x1F;
    if (n == 0) return typename W::T(v);
    uint32_t r = v & W::mask;
    bool cf = c.cf;
    c.of = ((r & W::sign) != 0) != cf;
    for (unsigned i = 0; i < n % (Bits + 1); i++) {
        const bool out = (r & 1) != 0;
        r = (r >> 1) | (cf ? W::sign : 0);
        cf = out;
    }
    c.cf = cf;
    return typename W::T(r);
}

inline uint8_t shl8(Cpu& c, uint8_t v, unsigned n) { return shlw<8>(c, v, n); }
inline uint16_t shl16(Cpu& c, uint16_t v, unsigned n) { return shlw<16>(c, v, n); }
inline uint8_t shr8(Cpu& c, uint8_t v, unsigned n) { return shrw<8>(c, v, n); }
inline uint16_t shr16(Cpu& c, uint16_t v, unsigned n) { return shrw<16>(c, v, n); }
inline uint8_t sar8(Cpu& c, uint8_t v, unsigned n) { return sarw<8>(c, v, n); }
inline uint16_t sar16(Cpu& c, uint16_t v, unsigned n) { return sarw<16>(c, v, n); }
inline uint8_t rol8(Cpu& c, uint8_t v, unsigned n) { return rolw<8>(c, v, n); }
inline uint16_t rol16(Cpu& c, uint16_t v, unsigned n) { return rolw<16>(c, v, n); }
inline uint8_t ror8(Cpu& c, uint8_t v, unsigned n) { return rorw<8>(c, v, n); }
inline uint16_t ror16(Cpu& c, uint16_t v, unsigned n) { return rorw<16>(c, v, n); }
inline uint8_t rcl8(Cpu& c, uint8_t v, unsigned n) { return rclw<8>(c, v, n); }
inline uint16_t rcl16(Cpu& c, uint16_t v, unsigned n) { return rclw<16>(c, v, n); }
inline uint8_t rcr8(Cpu& c, uint8_t v, unsigned n) { return rcrw<8>(c, v, n); }
inline uint16_t rcr16(Cpu& c, uint16_t v, unsigned n) { return rcrw<16>(c, v, n); }

inline void mul8(Machine&, Cpu& c, uint8_t v) {
    c.a.x = uint16_t(c.a.l * v);
    c.cf = c.of = c.a.h != 0;
}
inline void mul16(Machine&, Cpu& c, uint16_t v) {
    const uint32_t r = uint32_t(c.a.x) * v;
    c.a.x = uint16_t(r);
    c.d.x = uint16_t(r >> 16);
    c.cf = c.of = c.d.x != 0;
}
inline void imul8(Machine&, Cpu& c, uint8_t v) {
    const int16_t r = int16_t(int8_t(c.a.l) * int8_t(v));
    c.a.x = uint16_t(r);
    c.cf = c.of = r != int16_t(int8_t(c.a.l));
}
inline void imul16(Machine&, Cpu& c, uint16_t v) {
    const int32_t r = int32_t(int16_t(c.a.x)) * int16_t(v);
    c.a.x = uint16_t(r);
    c.d.x = uint16_t(uint32_t(r) >> 16);
    c.cf = c.of = r != int32_t(int16_t(c.a.x));
}
inline uint16_t imul16x(Cpu& c, uint16_t a, uint16_t b) {
    const int32_t r = int32_t(int16_t(a)) * int16_t(b);
    c.cf = c.of = r != int32_t(int16_t(r));
    return uint16_t(r);
}
inline void div8(Machine& m, Cpu& c, uint8_t v) {
    if (v == 0 || c.a.x / v > 0xFF) { m.divide_error(); return; }
    const uint16_t q = uint16_t(c.a.x / v), r = uint16_t(c.a.x % v);
    c.a.l = uint8_t(q);
    c.a.h = uint8_t(r);
}
inline void div16(Machine& m, Cpu& c, uint16_t v) {
    const uint32_t n = (uint32_t(c.d.x) << 16) | c.a.x;
    if (v == 0 || n / v > 0xFFFF) { m.divide_error(); return; }
    c.a.x = uint16_t(n / v);
    c.d.x = uint16_t(n % v);
}
inline void idiv8(Machine& m, Cpu& c, uint8_t v) {
    const int16_t n = int16_t(c.a.x);
    if (v == 0) { m.divide_error(); return; }
    const int q = n / int8_t(v), r = n % int8_t(v);
    if (q > 127 || q < -128) { m.divide_error(); return; }
    c.a.l = uint8_t(q);
    c.a.h = uint8_t(r);
}
inline void idiv16(Machine& m, Cpu& c, uint16_t v) {
    const int32_t n = int32_t((uint32_t(c.d.x) << 16) | c.a.x);
    if (v == 0) { m.divide_error(); return; }
    const int64_t q = int64_t(n) / int16_t(v), r = int64_t(n) % int16_t(v);
    if (q > 32767 || q < -32768) { m.divide_error(); return; }
    c.a.x = uint16_t(q);
    c.d.x = uint16_t(r);
}

}  // namespace nax
