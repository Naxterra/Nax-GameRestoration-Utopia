// OPL2 synthesis. Each of the nine channels has two operators (sine oscillators with an envelope); in FM mode the
// first one modulates the phase of the second, in additive mode both are heard. Envelope rates, key scaling,
// tremolo, vibrato, waveform selection, feedback and the five percussion voices follow the YM3812 data sheet.
// Phase modulation depths follow the chip's 10-bit sine index: a full-scale modulator moves the carrier by two
// cycles, and feedback 7 moves the modulator by one cycle.
#include "opl.h"

#include <algorithm>
#include <cmath>

namespace nax {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kUnitDb = 0.1875;  // one envelope step
const double kMultiple[16] = {0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 12, 12, 15, 15};
const double kKslDb[16] = {0, 9, 12, 13.875, 15, 16.125, 16.875, 17.625, 18, 18.75, 19.125, 19.5, 19.875, 20.25, 20.625, 21};

// Register offsets 00h-15h address 18 operators with gaps at 06h, 07h, 0Eh and 0Fh.
int slot_of(int offset) {
    static const int map[22] = {0, 1, 2, 3, 4, 5, -1, -1, 6, 7, 8, 9, 10, 11, -1, -1, 12, 13, 14, 15, 16, 17};
    return offset < 22 ? map[offset] : -1;
}
int first_slot(int ch) { return (ch / 3) * 6 + ch % 3; }

double attack_coef[64], decay_step[64];

struct Tables {
    Tables() {
        for (int r = 0; r < 64; r++) {
            // Times from the data sheet for the whole range: attack 2826 ms and decay 39280 ms at rate 4, halving
            // every four rates. The fraction r & 3 shortens the time within each group of four.
            const double scale = std::exp2((4.0 - (r & ~3)) / 4.0) * 4.0 / (4.0 + (r & 3));
            const double attack_time = 2.826 * scale, decay_time = 39.28 * scale;
            attack_coef[r] = r == 0 ? 0 : 6.24 / (attack_time * Opl2::kRate);
            decay_step[r] = r == 0 ? 0 : 512.0 / (decay_time * Opl2::kRate);
        }
    }
} tables;

double wave(int shape, double p) {
    const double s = std::sin(2 * kPi * p);
    switch (shape) {
    case 1: return s > 0 ? s : 0;
    case 2: return std::fabs(s);
    case 3: return std::fmod(p, 0.5) < 0.25 ? std::fabs(s) : 0;
    default: return s;
    }
}

double amplitude(double units) { return units >= 511 ? 0.0 : std::exp2(-units * kUnitDb / 6.0206); }

}  // namespace

Opl2::Opl2() = default;

void Opl2::set_key(int slot, bool on) {
    Operator& op = ops_[slot];
    if (on && !op.key) {
        op.stage = kAttack;
        op.phase = 0;
    } else if (!on && op.key && op.stage != kOff) {
        op.stage = kRelease;
    }
    op.key = on;
}

void Opl2::update_keys(int ch) {
    const int a = first_slot(ch), b = a + 3;
    bool ka = channels_[ch].key, kb = channels_[ch].key;
    if (rhythm_) {
        if (ch == 6) { ka |= (rhythm_keys_ & 0x10) != 0; kb |= (rhythm_keys_ & 0x10) != 0; }
        if (ch == 7) { ka |= (rhythm_keys_ & 0x01) != 0; kb |= (rhythm_keys_ & 0x08) != 0; }
        if (ch == 8) { ka |= (rhythm_keys_ & 0x04) != 0; kb |= (rhythm_keys_ & 0x02) != 0; }
    }
    set_key(a, ka);
    set_key(b, kb);
}

void Opl2::write(uint8_t reg, uint8_t v) {
    regs_[reg] = v;
    if (reg == 0x01) { wave_select_ = v & 0x20; return; }
    if (reg == 0x08) { note_select_ = v & 0x40; return; }
    if (reg == 0xBD) {
        deep_am_ = v & 0x80;
        deep_vib_ = v & 0x40;
        rhythm_ = v & 0x20;
        rhythm_keys_ = v & 0x1F;
        for (int ch = 6; ch < 9; ch++) update_keys(ch);
        return;
    }
    const int group = reg & 0xE0, offset = reg & 0x1F;
    if ((group >= 0x20 && group <= 0x80) || group == 0xE0) {
        const int slot = slot_of(offset);
        if (slot < 0) return;
        Operator& op = ops_[slot];
        switch (group) {
        case 0x20: op.am = v & 0x80; op.vib = v & 0x40; op.egt = v & 0x20; op.ksr = v & 0x10; op.mult = v & 0x0F; break;
        case 0x40: op.ksl = uint8_t(v >> 6); op.tl = v & 0x3F; break;
        case 0x60: op.ar = uint8_t(v >> 4); op.dr = v & 0x0F; break;
        case 0x80: op.sl = uint8_t(v >> 4); op.rr = v & 0x0F; break;
        case 0xE0: op.wave = v & 3; break;
        }
        return;
    }
    const int ch = reg & 0x0F;
    if (ch > 8) return;
    Channel& c = channels_[ch];
    if (group == 0xA0 && reg < 0xB0) {
        c.fnum = uint16_t((c.fnum & 0x300) | v);
    } else if (reg >= 0xB0 && reg <= 0xB8) {
        c.fnum = uint16_t((c.fnum & 0xFF) | ((v & 3) << 8));
        c.block = (v >> 2) & 7;
        c.key = v & 0x20;
        update_keys(ch);
    } else if (reg >= 0xC0 && reg <= 0xC8) {
        c.fb = (v >> 1) & 7;
        c.cnt = v & 1;
    }
}

int Opl2::rate(const Operator& op, const Channel& ch, int value) const {
    if (value == 0) return 0;
    const int keycode = (ch.block << 1) | ((note_select_ ? ch.fnum >> 8 : ch.fnum >> 9) & 1);
    return std::min(63, value * 4 + (op.ksr ? keycode : keycode >> 2));
}

void Opl2::envelope(Operator& op, const Channel& ch) {
    switch (op.stage) {
    case kAttack: {
        const int r = rate(op, ch, op.ar);
        if (r >= 60) op.env = 0;
        else if (r > 0) op.env -= (op.env + 1.0) * attack_coef[r];
        if (op.env <= 0) { op.env = 0; op.stage = kDecay; }
        break;
    }
    case kDecay: {
        const double level = op.sl == 15 ? 496 : op.sl * 16.0;
        op.env += decay_step[rate(op, ch, op.dr)];
        if (op.env >= level) { op.env = level; op.stage = kSustain; }
        break;
    }
    case kSustain:
        if (!op.egt) op.env += decay_step[rate(op, ch, op.rr)];
        break;
    case kRelease:
        op.env += decay_step[rate(op, ch, op.rr)];
        break;
    case kOff:
        break;
    }
    if (op.env >= 511) {
        op.env = 511;
        if (op.stage == kRelease || (op.stage == kSustain && !op.egt)) op.stage = kOff;
    }
}

// Advances one operator by one sample and returns its output for the given phase offset in cycles.
double Opl2::step(Operator& op, const Channel& ch, double vib, double trem, double modulation) {
    envelope(op, ch);
    double ksl = 0;
    if (op.ksl) {
        const double db = std::max(0.0, kKslDb[ch.fnum >> 6] - 6.0 * (7 - ch.block));
        ksl = (op.ksl == 3 ? db : (op.ksl == 1 ? db / 2 : db / 4)) / kUnitDb;
    }
    const double units = op.env + op.tl * 4.0 + ksl + (op.am ? trem : 0);
    double p = op.phase + modulation;
    p -= std::floor(p);
    const double out = op.stage == kOff ? 0.0 : wave(wave_select_ ? op.wave : 0, p) * amplitude(units);
    op.phase += ch.fnum * std::exp2(ch.block) / 1048576.0 * kMultiple[op.mult] * (op.vib ? vib : 1.0);
    op.phase -= std::floor(op.phase);
    return out;
}

void Opl2::generate(float* out, int count) {
    for (int n = 0; n < count; n++) {
        trem_phase_ += 3.7 / kRate;
        vib_phase_ += 6.1 / kRate;
        trem_phase_ -= std::floor(trem_phase_);
        vib_phase_ -= std::floor(vib_phase_);
        const double tri = trem_phase_ < 0.5 ? trem_phase_ * 2 : 2 - trem_phase_ * 2;
        const double trem = tri * (deep_am_ ? 4.8 : 1.0) / kUnitDb;
        const double vib = std::exp2((deep_vib_ ? 14.0 : 7.0) * std::sin(2 * kPi * vib_phase_) / 1200.0);
        noise_ = noise_ * 1103515245u + 12345u;
        const bool noise = (noise_ >> 30) & 1;

        double mix = 0;
        const int melodic = rhythm_ ? 6 : 9;
        for (int ch = 0; ch < 9; ch++) {
            Channel& c = channels_[ch];
            Operator& a = ops_[first_slot(ch)];
            Operator& b = ops_[first_slot(ch) + 3];
            if (ch < melodic || ch == 6) {
                const double fb = c.fb ? (a.out[0] + a.out[1]) * 0.5 * std::exp2(c.fb - 7.0) : 0.0;
                const double m = step(a, c, vib, trem, fb);
                a.out[1] = a.out[0];
                a.out[0] = m;
                const double v = c.cnt ? m + step(b, c, vib, trem, 0) : step(b, c, vib, trem, m * 2.0);
                mix += ch < melodic ? v : v * 2.0;  // the bass drum is heard at double level
                continue;
            }
            // Percussion on channels 7 and 8: hi-hat and snare share channel 7, tom-tom and cymbal channel 8.
            if (ch == 7) {
                const int hh = int(a.phase * 1024) & 1023, cym = int(ops_[17].phase * 1024) & 1023;
                const bool bit = (((hh >> 2) ^ (hh >> 7)) | ((cym >> 3) ^ (cym >> 5)) | ((hh >> 3) ^ (cym >> 5))) & 1;
                const double hh_phase = (bit ? (noise ? 0xD0 : 0x234) : (noise ? 0x34 : 0x2D0)) / 1024.0;
                const double sd_phase = (((hh >> 8) & 1) ? 0x200 : 0x100) / 1024.0 + (noise ? 0.25 : 0.0);
                mix += 2.0 * step(a, c, vib, trem, hh_phase - a.phase);
                mix += 2.0 * step(b, c, vib, trem, sd_phase - b.phase);
            } else {
                mix += 2.0 * step(a, c, vib, trem, 0);
                const int hh = int(ops_[13].phase * 1024) & 1023, cym = int(b.phase * 1024) & 1023;
                const bool bit = (((hh >> 2) ^ (hh >> 7)) | ((cym >> 3) ^ (cym >> 5)) | ((hh >> 3) ^ (cym >> 5))) & 1;
                mix += 2.0 * step(b, c, vib, trem, (bit ? 0x300 : 0x100) / 1024.0 - b.phase);
            }
        }
        out[n] = float(mix * 0.25);
    }
}

}  // namespace nax
