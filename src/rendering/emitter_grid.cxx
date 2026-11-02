#include "rendering/emitter_grid.hxx"

#include <algorithm>
#include <limits>

#include <glm/common.hpp>
#include <glm/vec3.hpp>

namespace {

    struct Span {
        glm::ivec3 first;
        glm::ivec3 last;
    };

    auto span_of(EmitterSphere const &emitter, glm::vec3 const &origin, float cell, glm::uvec3 const &dims) -> Span {
        auto const r = glm::vec3{emitter_reach(emitter.radius)};
        auto const top = glm::ivec3{dims} - 1;
        return {glm::clamp(glm::ivec3{glm::floor((emitter.centre - r - origin) / cell)}, glm::ivec3{0}, top),
                glm::clamp(glm::ivec3{glm::floor((emitter.centre + r - origin) / cell)}, glm::ivec3{0}, top)};
    }

    auto index_of(glm::ivec3 const &c, glm::uvec3 const &dims) -> std::uint32_t {
        return static_cast<std::uint32_t>(c.x) + dims.x * (static_cast<std::uint32_t>(c.y) + dims.y * static_cast<std::uint32_t>(c.z));
    }

}

auto build_emitter_grid(std::span<EmitterSphere const> emitters, std::uint32_t max_cells, std::uint32_t max_indices)
        -> EmitterGrid {
    EmitterGrid grid;
    if (emitters.empty()) {
        return grid;
    }

    auto low = glm::vec3{std::numeric_limits<float>::max()};
    auto high = glm::vec3{std::numeric_limits<float>::lowest()};
    auto cell = 4.0F;
    for (auto const &emitter: emitters) {
        auto const r = emitter_reach(emitter.radius);
        low = glm::min(low, emitter.centre - glm::vec3{r});
        high = glm::max(high, emitter.centre + glm::vec3{r});
        cell = std::max(cell, 2.0F * r);
    }

    for (auto attempt = 0; attempt < 40; ++attempt, cell *= 1.5F) {
        auto const dims = glm::max(glm::uvec3{glm::ceil((high - low) / cell)}, glm::uvec3{1});
        auto const cells = std::uint64_t{dims.x} * dims.y * dims.z;
        if (cells > max_cells) {
            continue;
        }

        std::vector<std::uint32_t> count(cells, 0U);
        auto total = std::uint64_t{0};
        for (auto const &emitter: emitters) {
            auto const [first, last] = span_of(emitter, low, cell, dims);
            for (auto z = first.z; z <= last.z; ++z) {
                for (auto y = first.y; y <= last.y; ++y) {
                    for (auto x = first.x; x <= last.x; ++x) {
                        ++count[index_of({x, y, z}, dims)];
                        ++total;
                    }
                }
            }
        }
        if (total > max_indices) {
            continue;
        }

        grid.valid = true;
        grid.origin = low;
        grid.cell = cell;
        grid.dims = dims;
        grid.count = std::move(count);
        grid.first.assign(cells, 0U);
        auto running = std::uint32_t{0};
        for (auto i = std::uint64_t{0}; i < cells; ++i) {
            grid.first[i] = running;
            running += grid.count[i];
        }
        grid.indices.assign(total, 0U);
        std::vector<std::uint32_t> fill(cells, 0U);
        for (auto e = std::uint32_t{0}; e < emitters.size(); ++e) {
            auto const [first, last] = span_of(emitters[e], low, cell, dims);
            for (auto z = first.z; z <= last.z; ++z) {
                for (auto y = first.y; y <= last.y; ++y) {
                    for (auto x = first.x; x <= last.x; ++x) {
                        auto const index = index_of({x, y, z}, dims);
                        grid.indices[grid.first[index] + fill[index]++] = e;
                    }
                }
            }
        }
        return grid;
    }
    return grid;
}
