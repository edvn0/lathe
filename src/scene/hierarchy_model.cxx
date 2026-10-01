#include "scene/hierarchy_model.hxx"

#include <algorithm>
#include <cctype>
#include <ranges>
#include <utility>

namespace {

    [[nodiscard]] auto to_lower(std::string_view text) -> std::string {
        std::string lower(text);
        std::ranges::transform(lower, lower.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return lower;
    }

} // namespace

auto HierarchyModel::rebuild(std::vector<Node> nodes, std::vector<std::string> group_labels) -> void {
    nodes_ = std::move(nodes);
    group_labels_ = std::move(group_labels);

    auto const node_count = static_cast<std::uint32_t>(nodes_.size());

    index_of_.clear();
    index_of_.reserve(nodes_.size());
    lower_names_.clear();
    lower_names_.reserve(nodes_.size());

    for (std::uint32_t node = 0; node < node_count; ++node) {
        index_of_.emplace(nodes_[node].entity, node);
        lower_names_.push_back(to_lower(nodes_[node].name));
    }

    // Each node's parent index, or node_count for a root.
    std::vector<std::uint32_t> parent_of(nodes_.size(), node_count);
    first_child_.assign(nodes_.size(), 0);
    child_count_.assign(nodes_.size(), 0);

    for (std::uint32_t node = 0; node < node_count; ++node) {
        auto const parent = nodes_[node].parent;
        if (parent == entt::null || parent == nodes_[node].entity) {
            continue;
        }
        if (auto const found = index_of_.find(parent); found != index_of_.end()) {
            parent_of[node] = found->second;
            ++child_count_[found->second];
        }
    }

    // Children laid out contiguously per parent, in node order.
    std::uint32_t running = 0;
    for (std::uint32_t node = 0; node < node_count; ++node) {
        first_child_[node] = running;
        running += child_count_[node];
    }

    children_.assign(running, 0);
    std::vector<std::uint32_t> filled(nodes_.size(), 0);

    roots_.clear();
    group_members_.assign(group_labels_.size(), {});

    for (std::uint32_t node = 0; node < node_count; ++node) {
        if (auto const parent = parent_of[node]; parent != node_count) {
            children_[first_child_[parent] + filled[parent]++] = node;
        } else if (auto const group = nodes_[node].group; group < group_members_.size()) {
            group_members_[group].push_back(node);
        } else {
            roots_.push_back(node);
        }
    }

    // A Parent cycle leaves its nodes unreachable from any root, so they are never listed (the walks below only
    // start from roots and group members), and a walk can't loop.

    recompute_matches();
    rows_dirty_ = true;
}

auto HierarchyModel::set_filter(std::string_view filter) -> void {
    auto lower = to_lower(filter);
    if (lower == filter_) {
        return;
    }

    filter_ = std::move(lower);
    recompute_matches();
    rows_dirty_ = true;
}

auto HierarchyModel::recompute_matches() -> void {
    auto const node_count = nodes_.size();
    self_match_.assign(node_count, filter_.empty());
    subtree_match_.assign(node_count, filter_.empty());

    filter_expanded_.clear();
    filter_expanded_groups_.clear();

    if (filter_.empty()) {
        return;
    }

    for (std::size_t node = 0; node < node_count; ++node) {
        self_match_[node] = lower_names_[node].find(filter_) != std::string::npos;
    }

    // Preorder from every listed start, then propagate matches up in reverse so children come before parents.
    std::vector<std::uint32_t> preorder;
    preorder.reserve(node_count);
    std::vector<std::uint32_t> stack;

    auto const walk = [&](std::uint32_t start) {
        stack.push_back(start);
        while (!stack.empty()) {
            auto const node = stack.back();
            stack.pop_back();
            preorder.push_back(node);
            for (std::uint32_t child = 0; child < child_count_[node]; ++child) {
                stack.push_back(children_[first_child_[node] + child]);
            }
        }
    };

    for (auto const root: roots_) {
        walk(root);
    }
    for (auto const &members: group_members_) {
        for (auto const member: members) {
            walk(member);
        }
    }

    for (auto const node: std::views::reverse(preorder)) {
        bool match = self_match_[node];
        for (std::uint32_t child = 0; child < child_count_[node] && !match; ++child) {
            match = subtree_match_[children_[first_child_[node] + child]];
        }
        subtree_match_[node] = match;
    }

    // Open every node with a matching descendant, and every group with a matching member.
    for (auto const node: preorder) {
        for (std::uint32_t child = 0; child < child_count_[node]; ++child) {
            if (subtree_match_[children_[first_child_[node] + child]]) {
                filter_expanded_.insert(nodes_[node].entity);
                break;
            }
        }
    }

    for (std::size_t group = 0; group < group_members_.size(); ++group) {
        if (std::ranges::any_of(group_members_[group], [&](std::uint32_t node) { return subtree_match_[node]; })) {
            filter_expanded_groups_.insert(group_labels_[group]);
        }
    }
}

auto HierarchyModel::set_expanded(entt::entity entity, bool expanded) -> void {
    auto &set = filtering() ? filter_expanded_ : expanded_;
    bool const changed = expanded ? set.insert(entity).second : set.erase(entity) != 0;
    rows_dirty_ |= changed;
}

auto HierarchyModel::set_group_expanded(std::uint32_t group, bool expanded) -> void {
    if (group >= group_labels_.size()) {
        return;
    }

    auto &set = filtering() ? filter_expanded_groups_ : expanded_groups_;
    bool const changed = expanded ? set.insert(group_labels_[group]).second : set.erase(group_labels_[group]) != 0;
    rows_dirty_ |= changed;
}

auto HierarchyModel::node_expanded(std::uint32_t node) const -> bool {
    auto const &set = filtering() ? filter_expanded_ : expanded_;
    return set.contains(nodes_[node].entity);
}

auto HierarchyModel::group_expanded(std::uint32_t group) const -> bool {
    auto const &set = filtering() ? filter_expanded_groups_ : expanded_groups_;
    return set.contains(group_labels_[group]);
}

auto HierarchyModel::rows() -> std::span<Row const> {
    if (rows_dirty_) {
        rebuild_rows();
        rows_dirty_ = false;
    }

    return rows_;
}

auto HierarchyModel::rebuild_rows() -> void {
    rows_.clear();

    // (node, depth), popped in sibling order because children are pushed in reverse.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;

    auto const emit_subtree = [&](std::uint32_t start, std::uint32_t start_depth) {
        stack.emplace_back(start, start_depth);

        while (!stack.empty()) {
            auto const [node, depth] = stack.back();
            stack.pop_back();

            if (!subtree_match_[node]) {
                continue;
            }

            bool const has_children = child_count_[node] != 0;
            bool const expanded = has_children && node_expanded(node);

            rows_.push_back(Row{
                    .entity = nodes_[node].entity,
                    .group = no_group,
                    .depth = depth,
                    .has_children = has_children,
                    .expanded = expanded,
            });

            if (expanded) {
                for (auto child = child_count_[node]; child-- > 0;) {
                    stack.emplace_back(children_[first_child_[node] + child], depth + 1);
                }
            }
        }
    };

    for (std::uint32_t group = 0; group < group_members_.size(); ++group) {
        auto const &members = group_members_[group];
        if (members.empty() ||
            std::ranges::none_of(members, [&](std::uint32_t node) { return subtree_match_[node]; })) {
            continue;
        }

        bool const expanded = group_expanded(group);
        rows_.push_back(Row{
                .entity = entt::null,
                .group = group,
                .depth = 0,
                .has_children = true,
                .expanded = expanded,
        });

        if (expanded) {
            for (auto const member: members) {
                emit_subtree(member, 1);
            }
        }
    }

    for (auto const root: roots_) {
        emit_subtree(root, 0);
    }
}

auto HierarchyModel::row_of(entt::entity entity) -> std::int64_t {
    if (entity == entt::null) {
        return -1;
    }

    auto const all_rows = rows();
    auto const found = std::ranges::find(all_rows, entity, &Row::entity);
    return found == all_rows.end() ? -1 : std::distance(all_rows.begin(), found);
}

auto HierarchyModel::matching_entities() const -> std::vector<entt::entity> {
    std::vector<entt::entity> matches;

    for (std::size_t node = 0; node < nodes_.size(); ++node) {
        if (self_match_[node]) {
            matches.push_back(nodes_[node].entity);
        }
    }

    return matches;
}

auto HierarchyModel::children_of(entt::entity entity) const -> std::vector<entt::entity> {
    std::vector<entt::entity> result;

    if (auto const found = index_of_.find(entity); found != index_of_.end()) {
        auto const node = found->second;
        result.reserve(child_count_[node]);
        for (std::uint32_t child = 0; child < child_count_[node]; ++child) {
            result.push_back(nodes_[children_[first_child_[node] + child]].entity);
        }
    }

    return result;
}
