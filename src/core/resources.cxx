#include "core/resources.hxx"

#include <atomic>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <mutex>
#include <utility>

namespace {
    std::atomic<std::shared_ptr<ResourceProvider const>> installed_provider;

    struct Recording {
        std::mutex mutex;
        bool active = false;
        std::map<std::string, std::vector<std::byte>> resources;
    };

    auto recording() -> Recording & {
        static Recording instance;

        return instance;
    }

    auto read_from_disk(DataPath const &path) -> std::optional<std::vector<std::byte>> {
        std::ifstream file{Paths::current().data_root() / path.logical(), std::ios::binary};

        if (!file) {
            return std::nullopt;
        }

        std::vector<char> const chars{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
        std::vector<std::byte> bytes(chars.size());

        std::ranges::transform(chars, bytes.begin(), [](char c) { return static_cast<std::byte>(c); });

        return bytes;
    }
}

auto install_resource_provider(std::shared_ptr<ResourceProvider const> provider) -> void {
    installed_provider.store(std::move(provider));
}

auto read_resource(DataPath const &path) -> std::optional<std::vector<std::byte>> {
    auto bytes = [&]() -> std::optional<std::vector<std::byte>> {
        if (auto const provider = installed_provider.load()) {
            if (auto found = provider->find(path.logical())) {
                return found;
            }
        }

        return read_from_disk(path);
    }();

    if (bytes) {
        auto &state = recording();
        std::scoped_lock const lock{state.mutex};

        if (state.active) {
            state.resources.insert_or_assign(path.logical(), *bytes);
        }
    }

    return bytes;
}

auto start_resource_recording() -> void {
    auto &state = recording();
    std::scoped_lock const lock{state.mutex};

    state.resources.clear();
    state.active = true;
}

auto finish_resource_recording() -> std::map<std::string, std::vector<std::byte>> {
    auto &state = recording();
    std::scoped_lock const lock{state.mutex};

    state.active = false;

    return std::exchange(state.resources, {});
}
