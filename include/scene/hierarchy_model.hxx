#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <entt/entity/entity.hpp>

class HierarchyModel {
public:
    static constexpr std::uint32_t no_group = std::numeric_limits<std::uint32_t>::max();

    struct Node {
        entt::entity entity = entt::null;

        entt::entity parent = entt::null;

        std::string name;

        std::uint32_t group = no_group;
    };

    struct Row {
        entt::entity entity = entt::null;

        std::uint32_t group = no_group;

        std::uint32_t depth = 0;
        bool has_children = false;
        bool expanded = false;
    };

    auto rebuild(std::vector<Node> nodes, std::vector<std::string> group_labels) -> void;

    auto set_filter(std::string_view filter) -> void;
    [[nodiscard]] auto filtering() const noexcept -> bool { return !filter_.empty(); }

    auto set_expanded(entt::entity entity, bool expanded) -> void;
    auto set_group_expanded(std::uint32_t group, bool expanded) -> void;

    [[nodiscard]] auto rows() -> std::span<Row const>;

    [[nodiscard]] auto row_of(entt::entity entity) -> std::int64_t;

    [[nodiscard]] auto matching_entities() const -> std::vector<entt::entity>;

    [[nodiscard]] auto children_of(entt::entity entity) const -> std::vector<entt::entity>;

    [[nodiscard]] auto node_count() const noexcept -> std::size_t { return nodes_.size(); }
    [[nodiscard]] auto group_label(std::uint32_t group) const -> std::string_view { return group_labels_[group]; }
    [[nodiscard]] auto group_size(std::uint32_t group) const -> std::size_t { return group_members_[group].size(); }

private:
    auto recompute_matches() -> void;
    auto rebuild_rows() -> void;

    [[nodiscard]] auto node_expanded(std::uint32_t node) const -> bool;
    [[nodiscard]] auto group_expanded(std::uint32_t group) const -> bool;

    std::vector<Node> nodes_;
    std::vector<std::string> lower_names_;
    std::unordered_map<entt::entity, std::uint32_t> index_of_;

    std::vector<std::uint32_t> first_child_;
    std::vector<std::uint32_t> child_count_;
    std::vector<std::uint32_t> children_;

    std::vector<std::uint32_t> roots_;
    std::vector<std::string> group_labels_;
    std::vector<std::vector<std::uint32_t>> group_members_;

    std::string filter_;
    std::vector<bool> self_match_;
    std::vector<bool> subtree_match_;

    std::unordered_set<entt::entity> expanded_;
    std::unordered_set<std::string> expanded_groups_;

    std::unordered_set<entt::entity> filter_expanded_;
    std::unordered_set<std::string> filter_expanded_groups_;

    std::vector<Row> rows_;
    bool rows_dirty_ = true;
};
