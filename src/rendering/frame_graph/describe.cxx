#include "rendering/frame_graph/describe.hxx"

#include <format>

#include "rendering/frame_graph/use_table.hxx"

namespace frame_graph {
    namespace {

        auto queue_name(LogicalQueue queue) -> char const * {
            return queue == LogicalQueue::graphics ? "graphics" : "compute";
        }

        auto resource_name(GraphDesc const &graph, std::uint32_t resource) -> std::string_view {
            return resource < graph.resources.size() ? std::string_view{graph.resources[resource].name}
                                                     : std::string_view{"<out of range>"};
        }

        auto append_barriers(std::string &text, GraphDesc const &graph, char const *label, BarrierSet const &set)
                -> void {
            for (auto const &barrier: set.images) {
                text += std::format("      {} image '{}' src({:#x},{:#x}) dst({:#x},{:#x}) layout {}->{} op {}\n",
                                    label, resource_name(graph, barrier.resource), barrier.src_stages,
                                    barrier.src_access, barrier.dst_stages, barrier.dst_access,
                                    static_cast<int>(barrier.old_layout), static_cast<int>(barrier.new_layout),
                                    static_cast<int>(barrier.op));
            }
            for (auto const &barrier: set.buffers) {
                text += std::format("      {} buffer '{}' src({:#x},{:#x}) dst({:#x},{:#x}) op {}\n", label,
                                    resource_name(graph, barrier.resource), barrier.src_stages, barrier.src_access,
                                    barrier.dst_stages, barrier.dst_access, static_cast<int>(barrier.op));
            }
            for (auto const &barrier: set.memory) {
                text += std::format("      {} memory src({:#x},{:#x}) dst({:#x},{:#x})\n", label, barrier.src_stages,
                                    barrier.src_access, barrier.dst_stages, barrier.dst_access);
            }
        }

    } // namespace

    auto describe(GraphDesc const &graph, CompiledGraph const &compiled, TransientPlan const *transients)
            -> std::string {
        auto text = std::string{};

        for (auto index = std::size_t{0}; index < compiled.batches.size(); ++index) {
            auto const &batch = compiled.batches[index];
            text += std::format("batch {} queue {} signal {} waits {}\n", index, queue_name(batch.queue),
                                batch.signal_index, batch.waits.size());
            for (auto const &wait: batch.waits) {
                text += std::format("    waits {} >= {}\n", queue_name(wait.queue), wait.signal_index);
            }
            append_barriers(text, graph, "acquire", batch.acquires);
            for (auto const &pass: batch.passes) {
                auto const &desc = graph.passes[pass.pass];
                text += std::format("    pass {}\n", desc.name);
                if (transients != nullptr) {
                    for (auto const &aliasing: transients->barriers) {
                        if (aliasing.pass == pass.pass) {
                            BarrierSet set;
                            set.memory.push_back(aliasing.barrier);
                            append_barriers(text, graph, "alias", set);
                        }
                    }
                }
                for (auto const &access: desc.accesses) {
                    auto const info = use_info(access.use, access.stages);
                    text += std::format("      use '{}' {} stages {:#x} access {:#x}{}\n",
                                        resource_name(graph, access.resource), static_cast<int>(access.use),
                                        info.stages, info.access, access.discard ? " discard" : "");
                }
                append_barriers(text, graph, "before", pass.before);
            }
            append_barriers(text, graph, "release", batch.releases);
            append_barriers(text, graph, "epilogue", batch.epilogue);
        }

        for (auto const &transfer: compiled.transfers) {
            text += std::format("transfer {} '{}' {} -> {} (release batch {}, acquire batch {})\n",
                                transfer.is_image ? "image" : "buffer", resource_name(graph, transfer.resource),
                                queue_name(transfer.from), queue_name(transfer.to), transfer.release_batch,
                                transfer.acquire_batch);
        }

        if (transients != nullptr) {
            text += std::format("transients: {} bytes in {} blocks ({} without aliasing)\n", transients->total_bytes,
                                transients->blocks.size(), transients->unaliased_bytes);
            for (auto const &placement: transients->placements) {
                text += std::format("    '{}' block {} offset {} size {}\n", resource_name(graph, placement.resource),
                                    placement.block, placement.offset, placement.size);
            }
        }

        return text;
    }

} // namespace frame_graph
