#include "scene/script_property.hxx"

#include <algorithm>
#include <array>
#include <cmath>

namespace {
    struct TypeName {
        ScriptPropertyType type;
        char const *name;
    };

    constexpr std::array type_names{
            TypeName{ScriptPropertyType::boolean, "boolean"}, TypeName{ScriptPropertyType::integer, "integer"},
            TypeName{ScriptPropertyType::number, "number"},   TypeName{ScriptPropertyType::string, "string"},
            TypeName{ScriptPropertyType::vec3, "vec3"},       TypeName{ScriptPropertyType::colour, "colour"},
            TypeName{ScriptPropertyType::choice, "choice"},   TypeName{ScriptPropertyType::entity, "entity"},
            TypeName{ScriptPropertyType::model, "model"},     TypeName{ScriptPropertyType::material, "material"},
            TypeName{ScriptPropertyType::texture, "texture"},
    };

    auto clamp_to(ScriptPropertyDesc const &desc, double value) -> double {
        if (!std::isfinite(value)) {
            return desc.has_range ? desc.min : 0.0;
        }

        return desc.has_range ? std::clamp(value, desc.min, desc.max) : value;
    }
}

auto script_type_name(ScriptPropertyType type) noexcept -> char const * {
    for (auto const &entry: type_names) {
        if (entry.type == type) {
            return entry.name;
        }
    }

    return "number";
}

auto script_type_from_name(std::string_view name, ScriptPropertyType &type) noexcept -> bool {
    // "color" for those who spell it that way.
    if (name == "color") {
        type = ScriptPropertyType::colour;

        return true;
    }

    for (auto const &entry: type_names) {
        if (name == entry.name) {
            type = entry.type;

            return true;
        }
    }

    return false;
}

auto script_value_matches(ScriptPropertyType type, ScriptValue const &value) noexcept -> bool {
    switch (type) {
        case ScriptPropertyType::boolean:
            return std::holds_alternative<bool>(value);
        case ScriptPropertyType::integer:
            return std::holds_alternative<std::int64_t>(value);
        case ScriptPropertyType::number:
            return std::holds_alternative<double>(value);
        case ScriptPropertyType::string:
        case ScriptPropertyType::choice:
        case ScriptPropertyType::model:
        case ScriptPropertyType::material:
        case ScriptPropertyType::texture:
            return std::holds_alternative<std::string>(value);
        case ScriptPropertyType::vec3:
            return std::holds_alternative<glm::vec3>(value);
        case ScriptPropertyType::colour:
            return std::holds_alternative<glm::vec4>(value);
        case ScriptPropertyType::entity:
            return std::holds_alternative<entt::entity>(value);
    }

    return false;
}

auto script_type_zero(ScriptPropertyType type) -> ScriptValue {
    switch (type) {
        case ScriptPropertyType::boolean:
            return false;
        case ScriptPropertyType::integer:
            return std::int64_t{0};
        case ScriptPropertyType::number:
            return 0.0;
        case ScriptPropertyType::string:
        case ScriptPropertyType::choice:
        case ScriptPropertyType::model:
        case ScriptPropertyType::material:
        case ScriptPropertyType::texture:
            return std::string{};
        case ScriptPropertyType::vec3:
            return glm::vec3{0.0F};
        case ScriptPropertyType::colour:
            return glm::vec4{1.0F};
        case ScriptPropertyType::entity:
            return entt::entity{entt::null};
    }

    return 0.0;
}

auto script_value_sanitised(ScriptPropertyDesc const &desc, ScriptValue value) -> ScriptValue {
    if (!script_value_matches(desc.type, value)) {
        return desc.default_value;
    }

    switch (desc.type) {
        case ScriptPropertyType::integer: {
            auto const clamped = clamp_to(desc, static_cast<double>(std::get<std::int64_t>(value)));

            return static_cast<std::int64_t>(std::llround(clamped));
        }
        case ScriptPropertyType::number:
            return clamp_to(desc, std::get<double>(value));
        case ScriptPropertyType::vec3: {
            auto vector = std::get<glm::vec3>(value);

            for (int axis = 0; axis < 3; ++axis) {
                vector[axis] = static_cast<float>(clamp_to(desc, static_cast<double>(vector[axis])));
            }

            return vector;
        }
        case ScriptPropertyType::colour: {
            auto colour = std::get<glm::vec4>(value);

            for (int channel = 0; channel < 4; ++channel) {
                if (!std::isfinite(colour[channel])) {
                    colour[channel] = 1.0F;
                }
            }

            return colour;
        }
        case ScriptPropertyType::choice: {
            auto const &text = std::get<std::string>(value);

            if (desc.choices.empty() || std::ranges::find(desc.choices, text) != desc.choices.end()) {
                return value;
            }

            return desc.choices.front();
        }
        default:
            return value;
    }
}
