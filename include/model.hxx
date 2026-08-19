#pragma once

#include <cstdint>

struct ModelHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return index != 0;
    }

    auto operator==(ModelHandle const &) const -> bool = default;
};

struct MeshHandle {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;

    [[nodiscard]]
    auto valid() const noexcept -> bool {
        return index != 0;
    }

    auto operator==(MeshHandle const &) const -> bool = default;
};
