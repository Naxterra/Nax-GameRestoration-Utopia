// A software FM synthesiser that follows the register interface of the Yamaha YM3812 (OPL2) on AdLib cards.
// It turns the register writes of the original music drivers into samples at the chip's own rate of 49716 Hz.
#pragma once

#include <cstdint>

namespace nax {

class Opl2 {
public:
    static constexpr int kRate = 49716;

    Opl2();
    void write(uint8_t reg, uint8_t value);
    void generate(float* out, int count);

private:
    enum Stage { kOff, kAttack, kDecay, kSustain, kRelease };

    struct Operator {
        uint8_t mult = 0, ksl = 0, tl = 0, ar = 0, dr = 0, sl = 0, rr = 0, wave = 0;
        bool am = false, vib = false, egt = false, ksr = false, key = false;
        double phase = 0, env = 511;
        Stage stage = kOff;
        double out[2] = {0, 0};
    };

    struct Channel {
        uint16_t fnum = 0;
        uint8_t block = 0, fb = 0;
        bool key = false, cnt = false;
    };

    void set_key(int op, bool on);
    void update_keys(int ch);
    double step(Operator& op, const Channel& ch, double lfo_vib, double lfo_trem, double modulation);
    void envelope(Operator& op, const Channel& ch);
    int rate(const Operator& op, const Channel& ch, int value) const;

    Operator ops_[18];
    Channel channels_[9];
    uint8_t regs_[256]{};
    bool wave_select_ = false, note_select_ = false, rhythm_ = false, deep_am_ = false, deep_vib_ = false;
    uint8_t rhythm_keys_ = 0;
    double trem_phase_ = 0, vib_phase_ = 0;
    uint32_t noise_ = 1;
};

}  // namespace nax
