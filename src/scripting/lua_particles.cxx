#include "scripting/lua_particles.hxx"

#include <array>
#include <cmath>
#include <expected>
#include <format>
#include <span>
#include <string_view>

namespace {
    using Problem = std::string;

    template<typename T>
    using Parsed = std::expected<T, Problem>;

    auto number(lua_State *state, int index, std::string_view key, double low, double high) -> Parsed<float> {
        if (lua_type(state, index) != LUA_TNUMBER) {
            return std::unexpected(std::format("'{}' must be a number", key));
        }

        auto const value = lua_tonumber(state, index);

        if (!std::isfinite(value) || value < low || value > high) {
            return std::unexpected(std::format("'{}' must be between {} and {}", key, low, high));
        }

        return static_cast<float>(value);
    }

    // {a, b, c [, d]}: `required` to `capacity` numbers in [low, high], in order. Missing optional ones keep `values`.
    auto numbers(lua_State *state, int index, std::string_view key, std::size_t required, std::span<float> values,
                 double low, double high) -> std::optional<Problem> {
        if (!lua_istable(state, index)) {
            return std::format("'{}' must be a table of {} numbers", key, required);
        }

        auto const length = lua_rawlen(state, index);

        if (length < required || length > values.size()) {
            return std::format("'{}' must have {} numbers", key,
                               required == values.size() ? std::to_string(required)
                                                         : std::format("{} or {}", required, values.size()));
        }

        for (auto position = std::size_t{0}; position < length; ++position) {
            lua_rawgeti(state, index, static_cast<lua_Integer>(position + 1));

            auto const element = number(state, -1, key, low, high);

            lua_pop(state, 1);

            if (!element) {
                return element.error();
            }

            values[position] = *element;
        }

        return std::nullopt;
    }

    auto shape_from(std::string_view name) -> std::optional<Components::ParticleShape> {
        using Components::ParticleShape;

        if (name == "point") {
            return ParticleShape::point;
        }
        if (name == "sphere") {
            return ParticleShape::sphere;
        }
        if (name == "box") {
            return ParticleShape::box;
        }
        if (name == "cone") {
            return ParticleShape::cone;
        }

        return std::nullopt;
    }

    constexpr auto colour_limit = 1000.0;

    auto apply_field(lua_State *state, std::string_view key, int value, Components::ParticleEmitter &emitter)
            -> std::optional<Problem> {
        auto const set = [&](float &target, double low, double high) -> std::optional<Problem> {
            auto const parsed = number(state, value, key, low, high);

            if (!parsed) {
                return parsed.error();
            }

            target = *parsed;

            return std::nullopt;
        };

        if (key == "count") {
            auto const parsed = number(state, value, key, 1.0, Components::ParticleEmitter::max_count);

            if (!parsed) {
                return parsed.error();
            }
            if (*parsed != std::floor(*parsed)) {
                return std::format("'{}' must be a whole number", key);
            }

            emitter.count = static_cast<std::uint32_t>(*parsed);

            return std::nullopt;
        }
        if (key == "rate") {
            return set(emitter.rate, 0.0, 1.0e6);
        }
        if (key == "lifetime") {
            return set(emitter.lifetime, 0.01, 3600.0);
        }
        if (key == "speed") {
            return set(emitter.speed, -1.0e4, 1.0e4);
        }
        if (key == "speed_variance") {
            return set(emitter.speed_variance, 0.0, 1.0);
        }
        if (key == "shape_size") {
            return set(emitter.shape_size, 0.0, 1.0e4);
        }
        if (key == "cone_degrees") {
            return set(emitter.cone_degrees, 0.0, 180.0);
        }
        if (key == "size_start") {
            return set(emitter.size_start, 0.0, 1.0e3);
        }
        if (key == "size_end") {
            return set(emitter.size_end, 0.0, 1.0e3);
        }
        if (key == "shape") {
            if (lua_type(state, value) != LUA_TSTRING) {
                return std::format("'{}' must be \"point\", \"sphere\", \"box\" or \"cone\"", key);
            }

            auto const shape = shape_from(lua_tostring(state, value));

            if (!shape) {
                return std::format("'{}' must be \"point\", \"sphere\", \"box\" or \"cone\"", key);
            }

            emitter.shape = *shape;

            return std::nullopt;
        }
        if (key == "emitting") {
            if (lua_type(state, value) != LUA_TBOOLEAN) {
                return std::format("'{}' must be true or false", key);
            }

            emitter.emitting = lua_toboolean(state, value) != 0;

            return std::nullopt;
        }
        if (key == "gravity") {
            auto values = std::array{emitter.gravity.x, emitter.gravity.y, emitter.gravity.z};

            if (auto const problem = numbers(state, value, key, 3, values, -1.0e4, 1.0e4)) {
                return problem;
            }

            emitter.gravity = {values[0], values[1], values[2]};

            return std::nullopt;
        }
        if (key == "colour_start" || key == "colour_end") {
            auto &colour = key == "colour_start" ? emitter.colour_start : emitter.colour_end;
            auto values = std::array{colour.x, colour.y, colour.z, colour.w};

            if (auto const problem = numbers(state, value, key, 3, values, 0.0, colour_limit)) {
                return problem;
            }
            if (values[3] > 1.0F) {
                return std::format("'{}' alpha must be between 0 and 1", key);
            }

            colour = {values[0], values[1], values[2], values[3]};

            return std::nullopt;
        }
        if (key == "material") {
            return std::nullopt;
        }

        return std::format("unknown particle field '{}'", key);
    }
}

auto read_particle_table(lua_State *state, int table, Components::ParticleEmitter &emitter)
        -> std::optional<std::string> {
    table = lua_absindex(state, table);

    if (!lua_istable(state, table)) {
        return "expected a table of particle fields";
    }

    auto const top = lua_gettop(state);
    auto updated = emitter;
    auto problem = std::optional<Problem>{};

    lua_pushnil(state);

    while (lua_next(state, table) != 0) {
        if (lua_type(state, -2) != LUA_TSTRING) {
            problem = "particle field names must be strings";
            break;
        }

        problem = apply_field(state, lua_tostring(state, -2), lua_gettop(state), updated);

        if (problem) {
            break;
        }

        lua_pop(state, 1);
    }

    lua_settop(state, top);

    if (!problem) {
        emitter = updated;
    }

    return problem;
}
