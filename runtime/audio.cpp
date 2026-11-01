#include "audio.h"

#include <windows.h>
#include <xaudio2.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace nax {

namespace {

constexpr int kBufferSamples = 512;
constexpr int kBuffersQueued = 4;
constexpr double kPitHz = 1193181.666;

struct VoiceCallback : IXAudio2VoiceCallback {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~VoiceCallback() { CloseHandle(event); }
    void STDMETHODCALLTYPE OnBufferEnd(void*) override { SetEvent(event); }
    void STDMETHODCALLTYPE OnStreamEnd() override {}
    void STDMETHODCALLTYPE OnVoiceProcessingPassEnd() override {}
    void STDMETHODCALLTYPE OnVoiceProcessingPassStart(UINT32) override {}
    void STDMETHODCALLTYPE OnBufferStart(void*) override {}
    void STDMETHODCALLTYPE OnLoopEnd(void*) override {}
    void STDMETHODCALLTYPE OnVoiceError(void*, HRESULT) override {}
};

}  // namespace

Audio::Audio() {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    freq_ = f.QuadPart;
    latency_ = freq_ * 60 / 1000;
    render_time_ = now() - latency_;
}

Audio::~Audio() { stop(); }

int64_t Audio::next_until(int64_t now_ticks, int samples) const {
    const int64_t span = int64_t(samples) * freq_ / Opl2::kRate;
    const int64_t target = render_time_ + span;
    const int64_t real = now_ticks - latency_;
    // After a stall of more than a quarter of a second the output catches up at once; render() then applies
    // the missed register writes silently instead of squeezing them into one buffer.
    if (real - target > freq_ / 4) return real;
    // Otherwise the sound card's clock is pulled gently towards the system clock: each buffer corrects an
    // eighth of the difference, at most a quarter of its length, so the output never runs ahead of the
    // register writes and never jumps.
    const int64_t correction = std::clamp((real - target) / 8, -span / 4, span / 4);
    return std::max(render_time_, target + correction);
}

int64_t Audio::now() const {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return v.QuadPart;
}

bool Audio::start(std::string* error) {
    render_time_ = now() - latency_;
    std::atomic<int> result{0};
    thread_ = std::thread([this, &result] {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        IXAudio2* xaudio = nullptr;
        IXAudio2MasteringVoice* master = nullptr;
        IXAudio2SourceVoice* voice = nullptr;
        VoiceCallback callback;
        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = 1;
        format.nSamplesPerSec = Opl2::kRate;
        format.wBitsPerSample = 16;
        format.nBlockAlign = 2;
        format.nAvgBytesPerSec = Opl2::kRate * 2;
        if (FAILED(XAudio2Create(&xaudio, 0, XAUDIO2_DEFAULT_PROCESSOR)) || FAILED(xaudio->CreateMasteringVoice(&master)) ||
            FAILED(xaudio->CreateSourceVoice(&voice, &format, 0, XAUDIO2_DEFAULT_FREQ_RATIO, &callback))) {
            result = -1;
            if (xaudio) xaudio->Release();
            CoUninitialize();
            return;
        }
        result = 1;
        voice->Start();
        std::vector<std::vector<int16_t>> buffers(kBuffersQueued + 1, std::vector<int16_t>(kBufferSamples));
        std::vector<float> samples(kBufferSamples);
        size_t next = 0;
        FILE* wav = record_path_.empty() ? nullptr : _wfopen(record_path_.c_str(), L"wb");
        uint32_t recorded = 0;
        auto header = [&] {
            const uint32_t bytes = recorded * 2;
            const uint32_t fields[] = {0x46464952, 36 + bytes, 0x45564157, 0x20746D66, 16, 0x00010001, uint32_t(Opl2::kRate),
                                       uint32_t(Opl2::kRate * 2), 0x00100002, 0x61746164, bytes};
            std::fseek(wav, 0, SEEK_SET);
            std::fwrite(fields, sizeof(fields), 1, wav);
            std::fseek(wav, 0, SEEK_END);
        };
        if (wav) header();
        while (!stop_) {
            XAUDIO2_VOICE_STATE state{};
            voice->GetState(&state, XAUDIO2_VOICE_NOSAMPLESPLAYED);
            for (UINT32 queued = state.BuffersQueued; queued < kBuffersQueued; queued++) {
                render(samples.data(), kBufferSamples, next_until(now(), kBufferSamples));
                auto& buf = buffers[next];
                next = (next + 1) % buffers.size();
                for (int i = 0; i < kBufferSamples; i++) buf[i] = int16_t(std::clamp(samples[i], -1.0f, 1.0f) * 32000.0f);
                if (wav) {
                    std::fwrite(buf.data(), 2, kBufferSamples, wav);
                    recorded += kBufferSamples;
                }
                XAUDIO2_BUFFER xb{};
                xb.AudioBytes = kBufferSamples * 2;
                xb.pAudioData = reinterpret_cast<const BYTE*>(buf.data());
                voice->SubmitSourceBuffer(&xb);
            }
            WaitForSingleObject(callback.event, 20);
        }
        if (wav) {
            header();
            std::fclose(wav);
        }
        voice->Stop();
        voice->DestroyVoice();
        master->DestroyVoice();
        xaudio->Release();
        CoUninitialize();
    });
    while (result == 0) Sleep(1);
    accepting_ = result > 0;
    if (result < 0) {
        thread_.join();
        *error = "XAudio2 could not open a sound device.";
        return false;
    }
    return true;
}

void Audio::stop() {
    accepting_ = false;
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

void Audio::push(const Event& e) {
    if (!accepting_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= kMaxEvents) {
        // The audio thread has stalled for a long time. Instead of dropping writes, which could leave a note
        // sounding forever, the queue is replaced by the complete current state of the chip and the speaker.
        events_.clear();
        for (int r = 1; r < 256; r++) events_.push_back({e.time, 0, uint8_t(r), shadow_[r]});
        events_.push_back({e.time, 1, speaker_bits_, speaker_reload_});
    }
    events_.push_back(e);
}

void Audio::opl_data(uint8_t value) {
    const int64_t t = now();
    switch (opl_index_) {
    case 0x02: t1_ = value; break;
    case 0x03: t2_ = value; break;
    case 0x04:
        if (value & 0x80) {
            status_ = 0;
        } else {
            if ((value & 1) && !(timer_control_ & 1)) { t1_start_ = t; status_reads_ = 0; }
            if ((value & 2) && !(timer_control_ & 2)) { t2_start_ = t; status_reads_ = 0; }
            timer_control_ = value;
        }
        break;
    }
    shadow_[opl_index_] = value;
    push({t, 0, opl_index_, value});
}

uint8_t Audio::opl_status() {
    // Each status read takes about a microsecond on a real card, and detection code counts on that delay.
    status_reads_++;
    const int64_t t = now();
    auto elapsed_us = [&](int64_t start) { return double(t - start) * 1e6 / double(freq_) + status_reads_; };
    if ((timer_control_ & 1) && !(timer_control_ & 0x40) && elapsed_us(t1_start_) >= (256 - t1_) * 80.0) status_ |= 0xC0;
    if ((timer_control_ & 2) && !(timer_control_ & 0x20) && elapsed_us(t2_start_) >= (256 - t2_) * 320.0) status_ |= 0xA0;
    return status_ | 0x06;
}

void Audio::speaker(bool gate, bool data, uint16_t reload) {
    speaker_bits_ = uint8_t((gate ? 1 : 0) | (data ? 2 : 0));
    speaker_reload_ = reload;
    push({now(), 1, speaker_bits_, reload});
}

void Audio::render(float* out, int count, int64_t until) {
    if (until < render_time_) until = render_time_;
    // A span longer than one buffer means the output fell behind. The writes in the missed time are applied
    // silently, so the chip's state is right, and only the last buffer length is heard.
    const int64_t span = int64_t(count) * freq_ / Opl2::kRate;
    if (until - render_time_ > 2 * span) {
        const int64_t skip_to = until - span;
        std::lock_guard<std::mutex> lock(mutex_);
        while (!events_.empty() && events_.front().time <= skip_to) {
            const Event& e = events_.front();
            if (e.kind == 0) {
                opl_.write(e.a, uint8_t(e.b));
            } else {
                speaker_on_ = (e.a & 3) == 3;
                speaker_level_ = e.a & 2;
                speaker_freq_ = kPitHz / (e.b ? e.b : 65536.0);
            }
            events_.pop_front();
        }
        render_time_ = skip_to;
    }
    const int64_t from = render_time_;
    std::deque<Event> due;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!events_.empty() && events_.front().time <= until) {
            due.push_back(events_.front());
            events_.pop_front();
        }
    }
    for (int i = 0; i < count; i++) {
        const int64_t t = from + (until - from) * (i + 1) / count;
        while (!due.empty() && due.front().time <= t) {
            const Event& e = due.front();
            if (e.kind == 0) {
                opl_.write(e.a, uint8_t(e.b));
            } else {
                const bool gate = e.a & 1, data = e.a & 2;
                speaker_on_ = gate && data;
                speaker_level_ = data;
                speaker_freq_ = kPitHz / (e.b ? e.b : 65536.0);
            }
            due.pop_front();
        }
        float fm = 0;
        opl_.generate(&fm, 1);
        // The speaker plays a square wave while the timer drives it, or follows the data bit directly.
        double level = 0;
        if (speaker_on_ && speaker_freq_ < Opl2::kRate / 2) {
            speaker_phase_ += speaker_freq_ / Opl2::kRate;
            speaker_phase_ -= std::floor(speaker_phase_);
            level = speaker_phase_ < 0.5 ? 0.15 : -0.15;
        } else if (!speaker_on_) {
            level = speaker_level_ ? 0.15 : 0.0;
        }
        speaker_dc_ += (level - speaker_dc_) * 0.001;  // remove the constant part, which would only click
        out[i] = fm + float(level - speaker_dc_);
    }
    render_time_ = until;
}

}  // namespace nax
