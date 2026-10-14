#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <entt/entity/entity.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// What a behaviour (a Lua file under assets/scripts/behaviours/) declares as editable, and what an entity
// stores for it. The schema comes from the script's `properties` table; an entity keeps only the values that differ
// from the defaults.

enum class ScriptPropertyType : std::uint8_t {
    boolean,
    integer,
    number,
    string,
    vec3,
    colour,
    choice,
    entity,
    // By registered asset name, so scenes keep working when handles change between runs.
    model,
    material,
    texture,
};

// boolean: bool; integer: int64; number: double; string and choice: string; vec3: glm::vec3; colour: glm::vec4;
// entity: entt::entity (null for none); model, material and texture: the registered asset name ("" for none).
using ScriptValue = std::variant<bool, std::int64_t, double, std::string, glm::vec3, glm::vec4, entt::entity>;

struct ScriptPropertyDesc {
    std::string name;
    ScriptPropertyType type = ScriptPropertyType::number;
    ScriptValue default_value{0.0};

    bool has_range = false;
    double min = 0.0;
    double max = 0.0;
    // Drag speed in the inspector; 0 picks one from the range.
    double step = 0.0;

    // The options of a `choice`.
    std::vector<std::string> choices;
    std::string tooltip;
};

struct ScriptSchema {
    std::string name;
    std::vector<ScriptPropertyDesc> properties;
    // Why the script could not be loaded, or empty. A script with an error has no properties.
    std::string error;

    [[nodiscard]] auto find(std::string_view property) const noexcept -> ScriptPropertyDesc const * {
        for (auto const &candidate: properties) {
            if (candidate.name == property) {
                return &candidate;
            }
        }

        return nullptr;
    }
};

struct ScriptPropertyValue {
    std::string name;
    ScriptValue value;
};

[[nodiscard]] auto script_type_name(ScriptPropertyType type) noexcept -> char const *;

// The type named by a script (`"number"`, `"vec3"`, ...); false when there is no such type.
[[nodiscard]] auto script_type_from_name(std::string_view name, ScriptPropertyType &type) noexcept -> bool;

// Whether `value` holds what `type` stores.
[[nodiscard]] auto script_value_matches(ScriptPropertyType type, ScriptValue const &value) noexcept -> bool;

// The value a property starts as when its script gives no default.
[[nodiscard]] auto script_type_zero(ScriptPropertyType type) -> ScriptValue;

// `value` forced into the property's range and options. Out-of-range numbers are clamped, a choice not in `choices`
// becomes the first option.
[[nodiscard]] auto script_value_sanitised(ScriptPropertyDesc const &desc, ScriptValue value) -> ScriptValue;
