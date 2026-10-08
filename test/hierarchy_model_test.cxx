#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "scene/hierarchy_model.hxx"

namespace {

    [[nodiscard]] auto entity(std::uint32_t id) -> entt::entity { return entt::entity{id}; }

    [[nodiscard]] auto node(std::uint32_t id, std::string name, entt::entity parent = entt::null,
                            std::uint32_t group = HierarchyModel::no_group) -> HierarchyModel::Node {
        return HierarchyModel::Node{.entity = entity(id), .parent = parent, .name = std::move(name), .group = group};
    }

    [[nodiscard]] auto layout(HierarchyModel &model) -> std::vector<std::pair<std::uint32_t, std::uint32_t>> {
        std::vector<std::pair<std::uint32_t, std::uint32_t>> result;
        for (auto const &row: model.rows()) {
            auto const id = row.entity == entt::null ? ~row.group : static_cast<std::uint32_t>(row.entity);
            result.emplace_back(id, row.depth);
        }
        return result;
    }

    using Layout = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

    [[nodiscard]] auto small_tree() -> std::vector<HierarchyModel::Node> {
        return {
                node(1, "Village"),         node(2, "House", entity(1)), node(3, "Well", entity(1)),
                node(4, "Lamp", entity(2)), node(5, "Player"),
        };
    }

}

TEST_CASE("Collapsed nodes list only the roots, in order") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});

    CHECK(layout(model) == Layout{{1, 0}, {5, 0}});
    CHECK(model.rows()[0].has_children);
    CHECK_FALSE(model.rows()[0].expanded);
    CHECK_FALSE(model.rows()[1].has_children);
}

TEST_CASE("Expanding a node lists its children one level deeper, in order") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});

    model.set_expanded(entity(1), true);
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {3, 1}, {5, 0}});

    model.set_expanded(entity(2), true);
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {4, 2}, {3, 1}, {5, 0}});

    model.set_expanded(entity(1), false);
    CHECK(layout(model) == Layout{{1, 0}, {5, 0}});

    model.set_expanded(entity(1), true);
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {4, 2}, {3, 1}, {5, 0}});
}

TEST_CASE("Expansion survives a rebuild") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});
    model.set_expanded(entity(1), true);

    auto nodes = small_tree();
    nodes.push_back(node(6, "Barn", entity(1)));
    model.rebuild(std::move(nodes), {});

    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {3, 1}, {6, 1}, {5, 0}});
}

TEST_CASE("Unknown, self and cyclic parents") {
    HierarchyModel model;
    model.rebuild(
            {
                    node(1, "Orphan", entity(99)),
                    node(2, "Self", entity(2)),
                    node(3, "Cycle A", entity(4)),
                    node(4, "Cycle B", entity(3)),
            },
            {});

    CHECK(layout(model) == Layout{{1, 0}, {2, 0}});
}

TEST_CASE("Group rows come first and list their members when expanded") {
    HierarchyModel model;
    model.rebuild(
            {
                    node(1, "Village"),
                    node(2, "bullet", entt::null, 0),
                    node(3, "bullet", entt::null, 0),
            },
            {"Bullets"});

    CHECK(model.group_label(0) == "Bullets");
    CHECK(model.group_size(0) == 2);
    CHECK(model.node_count() == 3);

    auto const group_row = ~std::uint32_t{0};
    CHECK(layout(model) == Layout{{group_row, 0}, {1, 0}});

    model.set_group_expanded(0, true);
    CHECK(layout(model) == Layout{{group_row, 0}, {2, 1}, {3, 1}, {1, 0}});
}

TEST_CASE("An empty group has no row") {
    HierarchyModel model;
    model.rebuild({node(1, "Village")}, {"Bullets"});

    CHECK(layout(model) == Layout{{1, 0}});
}

TEST_CASE("Filtering lists matches with their ancestors expanded") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});

    model.set_filter("LAMP");
    CHECK(model.filtering());
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {4, 2}});
    CHECK(model.matching_entities() == std::vector{entity(4)});

    model.set_expanded(entity(1), false);
    CHECK(layout(model) == Layout{{1, 0}});

    model.set_filter("lam");
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}, {4, 2}});

    model.set_filter("");
    CHECK_FALSE(model.filtering());
    CHECK(layout(model) == Layout{{1, 0}, {5, 0}});
    CHECK(model.matching_entities().size() == 5);
}

TEST_CASE("A matching parent lists its matching children only") {
    HierarchyModel model;
    model.rebuild(
            {
                    node(1, "Lamps"),
                    node(2, "Lamp A", entity(1)),
                    node(3, "Fence", entity(1)),
            },
            {});

    model.set_filter("lamp");
    CHECK(layout(model) == Layout{{1, 0}, {2, 1}});
    CHECK(model.matching_entities() == std::vector{entity(1), entity(2)});
}

TEST_CASE("Filtering opens groups with a matching member") {
    HierarchyModel model;
    model.rebuild(
            {
                    node(1, "Village"),
                    node(2, "bullet 7", entt::null, 0),
                    node(3, "bullet 8", entt::null, 0),
            },
            {"Bullets"});

    model.set_filter("8");
    CHECK(layout(model) == Layout{{~std::uint32_t{0}, 0}, {3, 1}});

    model.set_filter("village");
    CHECK(layout(model) == Layout{{1, 0}});
}

TEST_CASE("row_of finds the visible row of an entity") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});

    CHECK(model.row_of(entity(5)) == 1);
    CHECK(model.row_of(entity(4)) == -1);
    CHECK(model.row_of(entt::null) == -1);

    model.set_expanded(entity(1), true);
    model.set_expanded(entity(2), true);
    CHECK(model.row_of(entity(4)) == 2);
}

TEST_CASE("children_of returns the children in order") {
    HierarchyModel model;
    model.rebuild(small_tree(), {});

    CHECK(model.children_of(entity(1)) == std::vector{entity(2), entity(3)});
    CHECK(model.children_of(entity(4)).empty());
    CHECK(model.children_of(entity(42)).empty());
}

TEST_CASE("A large flat list rebuilds rows without walking collapsed subtrees") {
    HierarchyModel model;

    std::vector<HierarchyModel::Node> nodes;
    nodes.push_back(node(1, "Stress lights"));
    for (std::uint32_t id = 2; id < 35'002; ++id) {
        nodes.push_back(node(id, "stress_light_" + std::to_string(id), entity(1)));
    }
    model.rebuild(std::move(nodes), {});

    CHECK(model.rows().size() == 1);

    model.set_expanded(entity(1), true);
    CHECK(model.rows().size() == 35'001);

    model.set_filter("stress_light_35001");
    CHECK(layout(model) == Layout{{1, 0}, {35'001, 1}});
}
