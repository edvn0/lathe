#pragma once

#include <string>

#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/compiled_graph.hxx"
#include "rendering/frame_graph/frame_graph.hxx"

namespace frame_graph {

    // A readable dump of a compiled plan: per batch its queue, signal index and waits, then each pass with its
    // declared uses and the barriers derived before it, and the batch's ownership releases and epilogue; then the
    // ownership transfers and, if given, the transient placement. It replaces prose barrier documentation
    // (--frame-graph-dump) and reads well in a failing test's message.
    [[nodiscard]] auto describe(GraphDesc const &graph, CompiledGraph const &compiled,
                                TransientPlan const *transients = nullptr) -> std::string;

    // The same plan as a Graphviz digraph (--frame-graph-dot): one cluster per submission batch, coloured by queue,
    // passes as nodes, and an edge from the producer of every version a pass consumes, labelled with the resources it
    // carries. Dashed edges cross queues; dotted ones are timeline semaphore waits. Render with `dot -Tsvg`.
    [[nodiscard]] auto to_dot(GraphDesc const &graph, CompiledGraph const &compiled,
                              TransientPlan const *transients = nullptr) -> std::string;

} // namespace frame_graph
