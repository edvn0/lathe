#pragma once

#include <filesystem>
#include <mutex>
#include <unordered_set>
#include <vector>

class ShaderChangeQueue {
public:
    auto push(std::filesystem::path path) -> void {
        std::lock_guard const lock{mutex_};
        pending_.insert(std::move(path));
    }

    [[nodiscard]]
    auto drain() -> std::vector<std::filesystem::path> {
        std::lock_guard const lock{mutex_};

        auto result = std::vector<std::filesystem::path>(pending_.begin(), pending_.end());
        pending_.clear();

        return result;
    }

private:
    std::mutex mutex_;
    std::unordered_set<std::filesystem::path> pending_;
};
