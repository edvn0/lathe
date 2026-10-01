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

// The editor Hierarchy panel's entity tree, cached between frames and flattened into one row per line the panel
// can show, so the panel can clip to the visible rows instead of walking every entity each frame.
//
// rebuild() takes the tree; the rows follow from it, the search filter and which nodes are expanded, and are only
// recomputed when one of those changes.
class HierarchyModel {
public:
    static constexpr std::uint32_t no_group = std::numeric_limits<std::uint32_t>::max();

    struct Node {
        entt::entity entity = entt::null;

        // entt::null, or an entity that isn't one of the nodes, lists this node at the root.
        entt::entity parent = entt::null;

        std::string name;

        // A root node in a group is listed under the group's row rather than at the root.
        std::uint32_t group = no_group;
    };

    struct Row {
        // entt::null on a group row.
        entt::entity entity = entt::null;

        // Set on a group row only.
        std::uint32_t group = no_group;

        std::uint32_t depth = 0;
        bool has_children = false;
        bool expanded = false;
    };

    // `nodes` keep their order among siblings. Group rows come first, in `group_labels` order, then the roots.
    // Expansion state carries over for entities and groups that are still there.
    auto rebuild(std::vector<Node> nodes, std::vector<std::string> group_labels) -> void;

    // Case-insensitive substring match on node names. While it is non-empty, rows list only the nodes that match
    // and their ancestors, and every ancestor of a match starts expanded; collapsing one lasts until the filter
    // changes.
    auto set_filter(std::string_view filter) -> void;
    [[nodiscard]] auto filtering() const noexcept -> bool { return !filter_.empty(); }

    auto set_expanded(entt::entity entity, bool expanded) -> void;
    auto set_group_expanded(std::uint32_t group, bool expanded) -> void;

    // Recomputed here when stale, so it is not const.
    [[nodiscard]] auto rows() -> std::span<Row const>;

    // The row showing `entity`, or -1 when it is collapsed away or filtered out.
    [[nodiscard]] auto row_of(entt::entity entity) -> std::int64_t;

    // Every listed node whose own name matches the filter (every node without one), including collapsed and
    // grouped ones. Ctrl+A selects these.
    [[nodiscard]] auto matching_entities() const -> std::vector<entt::entity>;

    // `entity`'s children as of the last rebuild(), in order.
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

    // Children of node i are children_[first_child_[i], first_child_[i] + child_count_[i]).
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

    // Used instead of the two sets above while filtering: seeded with the ancestors of every match.
    std::unordered_set<entt::entity> filter_expanded_;
    std::unordered_set<std::string> filter_expanded_groups_;

    std::vector<Row> rows_;
    bool rows_dirty_ = true;
};
