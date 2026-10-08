#pragma once

#include <string>

#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    [[nodiscard]] auto describe(GraphDesc const &graph, CompiledGraph const &compiled,
                                TransientPlan const *transients = nullptr) -> std::string;

    [[nodiscard]] auto to_dot(GraphDesc const &graph, CompiledGraph const &compiled,
                              TransientPlan const *transients = nullptr) -> std::string;

}
