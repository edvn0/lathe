#include "rendering/effect_manifest.hxx"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <format>
#include <limits>
#include <ranges>
#include <span>

#include "core/json.hxx"

namespace {
    using Problem = std::string;

    template<typename T>
    using Parsed = std::expected<T, Problem>;

    auto is_identifier(std::string_view name) -> bool {
        if (name.empty() || name.size() > 64 || std::isdigit(static_cast<unsigned char>(name.front())) != 0) {
            return false;
        }
        return std::ranges::all_of(name, [](char c) {
            return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
        });
    }

    // Stays inside assets/shaders: relative, no "..", no backslashes, ends in .slang.
    auto is_shader_path(std::string_view path) -> bool {
        if (path.empty() || path.front() == '/' || path.find('\\') != std::string_view::npos ||
            !path.ends_with(".slang")) {
            return false;
        }
        return std::ranges::none_of(path | std::views::split('/'), [](auto part) {
            auto const view = std::string_view{part.begin(), part.end()};
            return view.empty() || view == "." || view == "..";
        });
    }

    auto param_type(std::string_view name) -> std::optional<EffectParamType> {
        if (name == "float") {
            return EffectParamType::float1;
        }
        if (name == "float2") {
            return EffectParamType::float2;
        }
        if (name == "float3") {
            return EffectParamType::float3;
        }
        if (name == "float4") {
            return EffectParamType::float4;
        }
        if (name == "int") {
            return EffectParamType::int1;
        }
        if (name == "uint") {
            return EffectParamType::uint1;
        }
        return std::nullopt;
    }

    auto numbers_of(JsonValue const &value) -> std::optional<std::vector<double>> {
        if (value.is_number()) {
            return std::vector{value.as_number()};
        }
        if (!value.is_array()) {
            return std::nullopt;
        }
        auto result = std::vector<double>{};
        for (auto const &item: value.items()) {
            if (!item.is_number()) {
                return std::nullopt;
            }
            result.push_back(item.as_number());
        }
        return result;
    }

    auto unknown_keys(JsonValue const &object, std::initializer_list<std::string_view> known) -> std::optional<Problem> {
        for (auto const &member: object.members()) {
            if (std::ranges::find(known, member.key) == known.end()) {
                return std::format("unknown key '{}'", member.key);
            }
        }
        return std::nullopt;
    }

    auto parse_param(JsonValue const &json) -> Parsed<EffectParam> {
        if (!json.is_object()) {
            return std::unexpected("a param must be an object");
        }
        if (auto const problem = unknown_keys(json, {"name", "type", "default", "min", "max"})) {
            return std::unexpected("param: " + *problem);
        }

        auto param = EffectParam{.name = json["name"].as_string()};
        if (!is_identifier(param.name)) {
            return std::unexpected(std::format("param name '{}' is not an identifier", param.name));
        }
        auto const type = param_type(json["type"].as_string());
        if (!type) {
            return std::unexpected(std::format("param '{}': type must be float, float2, float3, float4, int or uint",
                                               param.name));
        }
        param.type = *type;

        if (auto const *min = json.find("min")) {
            if (!min->is_number() || !std::isfinite(min->as_number())) {
                return std::unexpected(std::format("param '{}': min must be a number", param.name));
            }
            param.min = min->as_number();
        }
        if (auto const *max = json.find("max")) {
            if (!max->is_number() || !std::isfinite(max->as_number())) {
                return std::unexpected(std::format("param '{}': max must be a number", param.name));
            }
            param.max = max->as_number();
        }
        if (param.min > param.max) {
            return std::unexpected(std::format("param '{}': min is above max", param.name));
        }
        if (param.type == EffectParamType::uint1) {
            param.min = std::max(param.min, 0.0);
        }

        if (auto const *def = json.find("default")) {
            auto const values = numbers_of(*def);
            if (!values) {
                return std::unexpected(std::format("param '{}': default must be a number or array of numbers",
                                                   param.name));
            }
            auto checked = check_effect_value(param, *values);
            if (!checked) {
                return std::unexpected(std::format("param '{}': default {}", param.name, checked.error()));
            }
            param.value = *checked;
        } else {
            // 0 where that is allowed, else the nearest allowed value.
            for (auto &component: param.value) {
                component = std::clamp(0.0, param.min, param.max);
            }
        }
        return param;
    }

    auto parse_binding(JsonValue const &json) -> Parsed<EffectBinding> {
        if (!json.is_object()) {
            return std::unexpected("a binding must be an object");
        }
        if (auto const problem = unknown_keys(json, {"name", "image", "buffer", "source", "format", "replaces",
                                                      "elements"})) {
            return std::unexpected("binding: " + *problem);
        }

        auto binding = EffectBinding{.name = json["name"].as_string()};
        if (!is_identifier(binding.name)) {
            return std::unexpected(std::format("binding name '{}' is not an identifier", binding.name));
        }

        auto const has_image = json.find("image") != nullptr;
        auto const has_buffer = json.find("buffer") != nullptr;
        if (has_image == has_buffer) {
            return std::unexpected(std::format("binding '{}' must have exactly one of \"image\" and \"buffer\"",
                                               binding.name));
        }

        if (has_buffer) {
            auto buffer = EffectBufferBinding{};
            auto const access = json["buffer"].as_string();
            if (access == "read") {
                buffer.access = EffectBufferAccess::read;
            } else if (access == "write") {
                buffer.access = EffectBufferAccess::write;
            } else if (access == "read_write") {
                buffer.access = EffectBufferAccess::read_write;
            } else {
                return std::unexpected(std::format("binding '{}': buffer must be read, write or read_write",
                                                   binding.name));
            }
            if (auto const *elements = json.find("elements")) {
                auto const count = elements->as_number(-1.0);
                if (!elements->is_number() || count < 1.0 || count > max_effect_buffer_elements ||
                    count != std::floor(count)) {
                    return std::unexpected(std::format("binding '{}': elements must be a whole number from 1 to {}",
                                                       binding.name, max_effect_buffer_elements));
                }
                buffer.elements = static_cast<std::uint32_t>(count);
            }
            binding.what = buffer;
            return binding;
        }

        auto const direction = json["image"].as_string();
        if (direction == "in") {
            auto image = EffectImageIn{};
            if (auto const *source = json.find("source")) {
                auto const name = source->as_string();
                if (name == "scene_colour") {
                    image.source = EffectSource::scene_colour;
                } else if (name == "scene_depth") {
                    image.source = EffectSource::scene_depth;
                } else {
                    return std::unexpected(std::format("binding '{}': source must be scene_colour or scene_depth",
                                                       binding.name));
                }
            }
            binding.what = image;
        } else if (direction == "out") {
            auto image = EffectImageOut{};
            if (auto const *format = json.find("format")) {
                auto const name = format->as_string();
                if (name == "rgba16f") {
                    image.format = EffectFormat::rgba16f;
                } else if (name == "rgba8") {
                    image.format = EffectFormat::rgba8;
                } else if (name == "r32f") {
                    image.format = EffectFormat::r32f;
                } else {
                    return std::unexpected(std::format("binding '{}': format must be rgba16f, rgba8 or r32f",
                                                       binding.name));
                }
            }
            if (auto const *replaces = json.find("replaces")) {
                if (replaces->as_string() != "scene_colour") {
                    return std::unexpected(std::format("binding '{}': only scene_colour can be replaced",
                                                       binding.name));
                }
                image.replaces_scene_colour = true;
            }
            binding.what = image;
        } else {
            return std::unexpected(std::format("binding '{}': image must be \"in\" or \"out\"", binding.name));
        }
        return binding;
    }
}

auto EffectManifest::find_binding(std::string_view name) const noexcept -> EffectBinding const * {
    auto const found = std::ranges::find(bindings, name, &EffectBinding::name);
    return found != bindings.end() ? &*found : nullptr;
}

auto EffectManifest::find_param(std::string_view name) const noexcept -> EffectParam const * {
    auto const found = std::ranges::find(params, name, &EffectParam::name);
    return found != params.end() ? &*found : nullptr;
}

auto EffectManifest::replaces_scene_colour() const noexcept -> bool {
    return std::ranges::any_of(bindings, [](EffectBinding const &binding) {
        auto const *image = std::get_if<EffectImageOut>(&binding.what);
        return image != nullptr && image->replaces_scene_colour;
    });
}

auto EffectManifest::push_bytes() const noexcept -> std::size_t {
    auto offset = std::size_t{0};
    auto const add = [&](std::size_t align, std::size_t size) { offset = ((offset + align - 1U) & ~(align - 1U)) + size; };

    for (auto const &binding: bindings) {
        if (std::holds_alternative<EffectBufferBinding>(binding.what)) {
            add(8, 8);
        } else {
            add(4, 4);
        }
    }
    for (auto const &param: params) {
        add(push_alignment(param.type), 4 * component_count(param.type));
    }
    return offset;
}

auto check_effect_value(EffectParam const &param, std::span<double const> numbers)
        -> std::expected<std::array<double, 4>, std::string> {
    auto const wanted = component_count(param.type);
    if (numbers.size() != wanted) {
        return std::unexpected(std::format("needs {} number{}", wanted, wanted == 1 ? "" : "s"));
    }

    auto result = std::array<double, 4>{};
    for (auto index = std::size_t{0}; index < wanted; ++index) {
        auto const value = numbers[index];
        if (!std::isfinite(value) || value < param.min || value > param.max) {
            return std::unexpected(std::format("must be between {} and {}", param.min, param.max));
        }
        if (is_integer(param.type) && value != std::floor(value)) {
            return std::unexpected("must be a whole number");
        }
        if (param.type == EffectParamType::int1 && (value < -2147483648.0 || value > 2147483647.0)) {
            return std::unexpected("is out of the range of an int");
        }
        if (param.type == EffectParamType::uint1 && value > 4294967295.0) {
            return std::unexpected("is out of the range of a uint");
        }
        result[index] = value;
    }
    return result;
}

auto parse_effect_manifest(std::string_view json_text) -> std::expected<EffectManifest, std::string> {
    auto parsed = parse_json(json_text);
    if (!parsed) {
        return std::unexpected(std::format("not valid JSON at byte {}: {}", parsed.error().offset,
                                           parsed.error().message));
    }
    auto const &root = *parsed;
    if (!root.is_object()) {
        return std::unexpected("the manifest must be a JSON object");
    }
    if (auto const problem = unknown_keys(root, {"shader", "entry", "group_size", "bindings", "params", "dispatch"})) {
        return std::unexpected(*problem);
    }

    auto manifest = EffectManifest{.shader = root["shader"].as_string()};
    if (!is_shader_path(manifest.shader)) {
        return std::unexpected("shader must be a relative .slang path inside assets/shaders");
    }
    if (auto const *entry = root.find("entry")) {
        manifest.entry = entry->as_string();
        if (!is_identifier(manifest.entry)) {
            return std::unexpected("entry must be an identifier");
        }
    }
    if (auto const *group = root.find("group_size")) {
        auto const sizes = numbers_of(*group);
        if (!sizes || sizes->empty() || sizes->size() > 2) {
            return std::unexpected("group_size must be [x] or [x, y]");
        }
        for (auto index = std::size_t{0}; index < sizes->size(); ++index) {
            auto const size = (*sizes)[index];
            if (size < 1.0 || size > 1024.0 || size != std::floor(size)) {
                return std::unexpected("group_size entries must be whole numbers from 1 to 1024");
            }
            manifest.group_size[index] = static_cast<std::uint32_t>(size);
        }
        if (sizes->size() == 1) {
            manifest.group_size[1] = 1;
        }
    }

    if (auto const *bindings = root.find("bindings")) {
        if (!bindings->is_array()) {
            return std::unexpected("bindings must be an array");
        }
        for (auto const &item: bindings->items()) {
            auto binding = parse_binding(item);
            if (!binding) {
                return std::unexpected(binding.error());
            }
            manifest.bindings.push_back(std::move(*binding));
        }
    }
    if (auto const *params = root.find("params")) {
        if (!params->is_array()) {
            return std::unexpected("params must be an array");
        }
        for (auto const &item: params->items()) {
            auto param = parse_param(item);
            if (!param) {
                return std::unexpected(param.error());
            }
            manifest.params.push_back(std::move(*param));
        }
    }

    auto names = std::vector<std::string_view>{};
    for (auto const &binding: manifest.bindings) {
        names.push_back(binding.name);
    }
    for (auto const &param: manifest.params) {
        names.push_back(param.name);
    }
    std::ranges::sort(names);
    if (auto const duplicate = std::ranges::adjacent_find(names); duplicate != names.end()) {
        return std::unexpected(std::format("'{}' is declared twice", *duplicate));
    }
    if (manifest.bindings.empty() && manifest.params.empty()) {
        return std::unexpected("an effect needs at least one binding or param");
    }
    if (std::ranges::count_if(manifest.bindings, [](EffectBinding const &binding) {
            auto const *image = std::get_if<EffectImageOut>(&binding.what);
            return image != nullptr && image->replaces_scene_colour;
        }) > 1) {
        return std::unexpected("only one binding can replace the scene colour");
    }
    if (manifest.push_bytes() > max_effect_push_bytes) {
        return std::unexpected(std::format("the push constants are {} bytes, the most is {}", manifest.push_bytes(),
                                           max_effect_push_bytes));
    }

    auto const *dispatch = root.find("dispatch");
    if (dispatch == nullptr || !dispatch->is_object() || dispatch->members().size() != 1) {
        return std::unexpected("dispatch must be an object with one of per_pixel_of, threads or elements_of");
    }
    auto const &rule = dispatch->members().front();
    if (rule.key == "per_pixel_of") {
        auto const *target = manifest.find_binding(rule.value.as_string());
        if (target == nullptr || std::holds_alternative<EffectBufferBinding>(target->what)) {
            return std::unexpected("dispatch.per_pixel_of must name an image binding");
        }
        manifest.dispatch = {.kind = EffectDispatch::Kind::per_pixel, .of = target->name};
    } else if (rule.key == "elements_of") {
        auto const *target = manifest.find_binding(rule.value.as_string());
        if (target == nullptr || !std::holds_alternative<EffectBufferBinding>(target->what)) {
            return std::unexpected("dispatch.elements_of must name a buffer binding");
        }
        manifest.dispatch = {.kind = EffectDispatch::Kind::elements, .of = target->name};
    } else if (rule.key == "threads") {
        auto const count = rule.value.as_number(-1.0);
        if (!rule.value.is_number() || count < 1.0 || count > max_effect_threads || count != std::floor(count)) {
            return std::unexpected(std::format("dispatch.threads must be a whole number from 1 to {}",
                                               max_effect_threads));
        }
        manifest.dispatch = {.kind = EffectDispatch::Kind::threads, .threads = static_cast<std::uint32_t>(count)};
    } else {
        return std::unexpected(std::format("unknown dispatch rule '{}'", rule.key));
    }

    return manifest;
}
