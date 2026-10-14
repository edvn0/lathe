#pragma once

#include <chrono>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <vector>

// Paths a FileChangeWatcher saw change, handed from its thread to whoever polls (shader reload, behaviour reload).
// Editors save in several steps and the watcher reports each, so a path is only handed out once it has been quiet for
// `debounce`; until then it stays pending and a new event for it restarts the wait.
class FileChangeQueue {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::milliseconds default_debounce{100};

    explicit FileChangeQueue(std::chrono::milliseconds debounce = default_debounce) noexcept : debounce_(debounce) {}

    auto push(std::filesystem::path path, Clock::time_point now = Clock::now()) -> void {
        std::lock_guard const lock{mutex_};
        pending_.insert_or_assign(std::move(path), now);
    }

    // The paths that have been quiet for the debounce time as of `now`; they are removed from the queue.
    [[nodiscard]]
    auto drain(Clock::time_point now = Clock::now()) -> std::vector<std::filesystem::path> {
        std::lock_guard const lock{mutex_};

        std::vector<std::filesystem::path> result;

        for (auto it = pending_.begin(); it != pending_.end();) {
            if (now - it->second >= debounce_) {
                result.push_back(it->first);
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }

        return result;
    }

private:
    std::mutex mutex_;
    std::chrono::milliseconds debounce_;
    std::unordered_map<std::filesystem::path, Clock::time_point> pending_;
};
