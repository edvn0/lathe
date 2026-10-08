#include "animation/joint_remap.hxx"
#include "animation/humanoid.hxx"

#include <algorithm>
#include <array>
#include <cctype>
#include <unordered_map>

namespace Animation {
    namespace {
        auto lower_alnum(std::string_view name) -> std::string {
            std::string out;
            for (char const c: name) {
                if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
                    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                }
            }
            return out;
        }

        auto strip_prefix(std::string &text, std::string_view prefix) -> bool {
            if (text.starts_with(prefix)) {
                text.erase(0, prefix.size());
                return true;
            }
            return false;
        }
    } // namespace

    auto canonical_joint_name(std::string_view const name) -> std::string {
        // Side markers: "Left"/"Right" prefix, or a trailing ".L"/"_R"/"_l" style suffix in the raw name.
        char side = 0;
        std::string_view raw = name;
        if (raw.size() > 2 && (raw[raw.size() - 2] == '.' || raw[raw.size() - 2] == '_')) {
            auto const last = static_cast<char>(std::tolower(static_cast<unsigned char>(raw.back())));
            if (last == 'l' || last == 'r') {
                side = last;
                raw.remove_suffix(2);
            }
        }

        auto text = lower_alnum(raw);
        strip_prefix(text, "mixamorig");
        if (strip_prefix(text, "left")) {
            side = 'l';
        } else if (strip_prefix(text, "right")) {
            side = 'r';
        }

        static std::unordered_map<std::string, std::string> const table{
                {"hips", "pelvis"},     {"pelvis", "pelvis"},   {"root", "pelvis"},     {"spine", "spine"},
                {"spine2", "chest"},    {"chest", "chest"},     {"upperchest", "chest"}, {"head", "head"},
                {"arm", "upperarm"},    {"uparm", "upperarm"},  {"upperarm", "upperarm"}, {"forearm", "forearm"},
                {"lowerarm", "forearm"}, {"upleg", "thigh"},    {"upperleg", "thigh"},  {"thigh", "thigh"},
                {"leg", "shin"},        {"lowerleg", "shin"},   {"calf", "shin"},       {"shin", "shin"},
                {"foot", "foot"},
        };
        if (auto const found = table.find(text); found != table.end()) {
            text = found->second;
        }
        if (side != 0) {
            text.push_back(side);
        }
        return text;
    }

    auto remap_joints(Skeleton const &source, std::span<std::string const> const target_names) -> JointRemap {
        JointRemap remap;
        remap.source_to_target.assign(source.joint_count(), no_joint);
        remap.target_to_source.assign(target_names.size(), no_joint);

        std::vector<std::string> target_keys;
        target_keys.reserve(target_names.size());
        for (auto const &name: target_names) {
            target_keys.push_back(canonical_joint_name(name));
        }

        for (std::size_t s = 0; s < source.joint_count(); ++s) {
            auto const key = canonical_joint_name(source.names()[s]);
            for (std::size_t t = 0; t < target_keys.size(); ++t) {
                if (target_keys[t] == key && remap.target_to_source[t] == no_joint) {
                    remap.target_to_source[t] = static_cast<std::int32_t>(s);
                    remap.source_to_target[s] = static_cast<std::int32_t>(t);
                    ++remap.mapped_count;
                    break;
                }
            }
        }
        return remap;
    }

    auto humanoid_joint_names() -> std::span<std::string const> {
        static auto const names = [] {
            auto const skeleton = Humanoid::make_skeleton();
            return std::vector<std::string>{skeleton.names().begin(), skeleton.names().end()};
        }();
        return names;
    }

    auto remap_to_humanoid(Skeleton const &source) -> JointRemap { return remap_joints(source, humanoid_joint_names()); }
} // namespace Animation
