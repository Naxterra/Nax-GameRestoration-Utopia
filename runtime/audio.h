// Sound output. The game thread reports AdLib register writes and PC speaker changes with a time stamp, and an
// audio thread turns them into samples a fixed latency later, so the music keeps the driver's own timing.
#pragma once

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "opl.h"

namespace nax {

class Audio {
public:
    Audio();
    ~Audio();
    bool start(std::string* error);
    // Also writes everything that is played to a WAV file, for checking the output without listening.
    void record(const std::filesystem::path& wav) { record_path_ = wav; }
    void stop();

    // Called on the game thread from the port handlers.
    void opl_address(uint8_t index) { opl_index_ = index; }
    void opl_data(uint8_t value);
    uint8_t opl_status();
    void speaker(bool gate, bool data, uint16_t reload);

    // Renders samples for the given time span; public so the tests can drive it without a sound device.
    void render(float* out, int count, int64_t until);
    int64_t now() const;
    // The end of the next buffer: one buffer length after the last one, or the current time after a long stall.
    int64_t next_until(int64_t now_ticks, int samples) const;
    int64_t render_time() const { return render_time_; }
    int64_t latency() const { return latency_; }
    size_t queued_events() {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }
    // Returns the last value queued for an OPL register, for the tests.
    bool queued_register(uint8_t reg, uint8_t* value) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = events_.rbegin(); it != events_.rend(); ++it)
            if (it->kind == 0 && it->a == reg) { *value = uint8_t(it->b); return true; }
        return false;
    }
    // Lets the tests queue events without a sound device.
    void accept_events_for_test() { accepting_ = true; }
    static constexpr size_t kMaxEvents = 65536;

private:
    struct Event {
        int64_t time;
        uint8_t kind;  // 0: OPL register write, 1: speaker change
        uint8_t a;
        uint16_t b;
    };
    void push(const Event& e);
    void thread_main();

    std::mutex mutex_;
    std::deque<Event> events_;
    Opl2 opl_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> accepting_{false};  // events are queued only while something renders them
    int64_t freq_ = 1;
    int64_t latency_ = 0;
    int64_t render_time_ = 0;
    std::filesystem::path record_path_;

    // OPL timers, used by the drivers to detect the card. Only the game thread touches these.
    uint8_t opl_index_ = 0, t1_ = 0, t2_ = 0, timer_control_ = 0, status_ = 0;
    uint8_t shadow_[256]{};  // the last value written to each register, to rebuild the state after an overflow
    uint8_t speaker_bits_ = 0;
    uint16_t speaker_reload_ = 0;
    int64_t t1_start_ = 0, t2_start_ = 0;
    int status_reads_ = 0;

    // PC speaker, as seen by the audio thread.
    bool speaker_on_ = false, speaker_level_ = false;
    double speaker_freq_ = 0, speaker_phase_ = 0, speaker_dc_ = 0;
};

}  // namespace nax
