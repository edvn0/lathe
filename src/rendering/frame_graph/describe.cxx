#include "rendering/frame_graph/describe.hxx"

#include <algorithm>
#include <format>
#include <map>
#include <set>

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

    namespace {

        struct QueueStyle {
            char const *fill;
            char const *border;
        };

        auto queue_style(LogicalQueue queue) -> QueueStyle {
            return queue == LogicalQueue::graphics ? QueueStyle{"#EAF1FB", "#1A73E8"}
                                                   : QueueStyle{"#E9F5EC", "#188038"};
        }

        auto kind_color(ResourceKind kind) -> char const * {
            switch (kind) {
                case ResourceKind::image:
                    return "#3C4043";
                case ResourceKind::buffer:
                    return "#B06000";
                case ResourceKind::token:
                    return "#80868B";
            }
            return "#3C4043";
        }

        auto pass_type_name(PassType type) -> char const * {
            switch (type) {
                case PassType::raster:
                    return "raster";
                case PassType::compute:
                    return "compute";
                case PassType::transfer:
                    return "transfer";
            }
            return "?";
        }

        auto escape(std::string_view text, bool html) -> std::string {
            auto out = std::string{};
            for (auto const c: text) {
                switch (c) {
                    case '&':
                        out += html ? "&amp;" : "&";
                        break;
                    case '<':
                        out += html ? "&lt;" : "<";
                        break;
                    case '>':
                        out += html ? "&gt;" : ">";
                        break;
                    case '"':
                        out += html ? "&quot;" : "\\\"";
                        break;
                    case '\\':
                        out += html ? "\\" : "\\\\";
                        break;
                    default:
                        out += c;
                }
            }
            return out;
        }

        auto mebibytes(std::uint64_t bytes) -> double { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

        auto barrier_count(BarrierSet const &set) -> std::size_t {
            return set.images.size() + set.buffers.size() + set.memory.size();
        }

    } // namespace

    auto to_dot(GraphDesc const &graph, CompiledGraph const &compiled, TransientPlan const *transients)
            -> std::string {
        auto text = std::string{};
        text += "digraph frame_graph {\n"
                "    graph [rankdir=LR, compound=true, splines=spline, nodesep=0.35, ranksep=0.7, pad=0.4,\n"
                "           fontname=\"Helvetica\", fontsize=11, labelloc=b, labeljust=l, bgcolor=\"white\"];\n"
                "    node [fontname=\"Helvetica\", fontsize=11, shape=box, style=\"rounded,filled\", fillcolor=white,\n"
                "          color=\"#5F6368\", penwidth=1.2, margin=\"0.16,0.08\"];\n"
                "    edge [fontname=\"Helvetica\", fontsize=9, color=\"#5F6368\", fontcolor=\"#3C4043\", arrowsize=0.7];\n\n";

        // Persistent imports (everything but the swapchain) are listed on the pass that reads them instead of drawn as
        // nodes, which would bury the pass-to-pass dependencies under fan-out edges.
        auto import_inputs = std::map<std::uint32_t, std::set<std::string>>{};
        for (auto const &batch: compiled.batches) {
            for (auto const &pass: batch.passes) {
                for (auto const &access: graph.passes[pass.pass].accesses) {
                    if (access.resource >= graph.producers.size() || access.version >= graph.producers[access.resource].size()) {
                        continue;
                    }
                    auto const &resource = graph.resources[access.resource];
                    if (graph.producers[access.resource][access.version] < 0 && resource.imported &&
                        !resource.swapchain) {
                        import_inputs[pass.pass].insert(escape(resource.name, true));
                    }
                }
            }
        }

        auto first_pass = std::vector<std::int64_t>(compiled.batches.size(), -1);
        auto last_pass = std::vector<std::int64_t>(compiled.batches.size(), -1);
        auto live = std::set<std::uint32_t>{};
        auto passes = std::size_t{0};

        for (auto index = std::size_t{0}; index < compiled.batches.size(); ++index) {
            auto const &batch = compiled.batches[index];
            auto const style = queue_style(batch.queue);
            text += std::format("    subgraph cluster_batch_{} {{\n", index);
            text += std::format("        label=<<b>Batch {}</b>  {}  signal {}>;\n", index, queue_name(batch.queue),
                                batch.signal_index);
            text += std::format("        style=\"rounded,filled\"; fillcolor=\"{}\"; color=\"{}\"; penwidth=1.5;\n",
                                style.fill, style.border);
            text += "        fontname=\"Helvetica\"; fontsize=12; fontcolor=\"#202124\"; margin=14;\n";
            if (batch.passes.empty()) {
                text += std::format("        batch_{} [label=\"(no passes)\", shape=plaintext, style=\"\", "
                                    "fontcolor=\"#80868B\"];\n",
                                    index);
            }
            for (auto const &pass: batch.passes) {
                auto const &desc = graph.passes[pass.pass];
                live.insert(pass.pass);
                ++passes;
                first_pass[index] = first_pass[index] < 0 ? pass.pass : first_pass[index];
                last_pass[index] = pass.pass;
                auto detail = std::format("{} \xC2\xB7 {}", pass_type_name(desc.type), queue_name(batch.queue));
                if (auto const barriers = barrier_count(pass.before); barriers > 0) {
                    detail += std::format(" \xC2\xB7 {} barrier{}", barriers, barriers == 1 ? "" : "s");
                }
                auto inputs = std::string{};
                for (auto const &name: import_inputs[pass.pass]) {
                    inputs += inputs.empty() ? "" : ", ";
                    inputs += name;
                }
                text += std::format("        p{} [color=\"{}\", label=<<table border=\"0\" cellborder=\"0\" "
                                    "cellspacing=\"0\" cellpadding=\"1\"><tr><td><b>{}</b></td></tr><tr><td>"
                                    "<font point-size=\"9\" color=\"#5F6368\">{}</font></td></tr>{}</table>>];\n",
                                    pass.pass, style.border, escape(desc.name, true), detail,
                                    inputs.empty() ? std::string{}
                                                   : std::format("<tr><td><font point-size=\"8\" "
                                                                 "color=\"#80868B\">imports: {}</font></td></tr>",
                                                                 inputs));
            }
            text += "    }\n\n";
        }

        // One edge per (producer, consumer): every resource version the consumer reads or writes over, merged.
        using Endpoint = std::pair<std::string, std::string>;
        auto carried = std::map<Endpoint, std::set<std::uint32_t>>{};
        auto imports = std::set<std::uint32_t>{};
        auto pass_queue = [&](std::uint32_t pass) {
            return pass < compiled.pass_queue.size() ? compiled.pass_queue[pass] : LogicalQueue::graphics;
        };
        auto crossing = std::set<Endpoint>{};

        for (auto const consumer: live) {
            for (auto const &access: graph.passes[consumer].accesses) {
                if (access.resource >= graph.producers.size()) {
                    continue;
                }
                auto const &versions = graph.producers[access.resource];
                auto const producer = access.version < versions.size() ? versions[access.version] : std::int64_t{-1};
                auto from = std::string{};
                if (producer < 0) {
                    if (!graph.resources[access.resource].imported || !graph.resources[access.resource].swapchain) {
                        continue;
                    }
                    imports.insert(access.resource);
                    from = std::format("r{}", access.resource);
                } else if (static_cast<std::uint32_t>(producer) == consumer ||
                           !live.contains(static_cast<std::uint32_t>(producer))) {
                    continue;
                } else {
                    from = std::format("p{}", producer);
                }
                auto const key = Endpoint{from, std::format("p{}", consumer)};
                carried[key].insert(access.resource);
                if (producer >= 0 && pass_queue(static_cast<std::uint32_t>(producer)) != pass_queue(consumer)) {
                    crossing.insert(key);
                }
            }
        }

        for (auto const resource: imports) {
            auto const &desc = graph.resources[resource];
            text += std::format("    r{} [shape=cylinder, style=\"filled\", fillcolor=\"#F1F3F4\", "
                                "color=\"#80868B\", label=\"{}\"];\n",
                                resource, escape(desc.name, false));
        }

        auto presented = std::vector<std::int64_t>{};
        for (auto resource = std::size_t{0}; resource < graph.resources.size(); ++resource) {
            if (graph.resources[resource].swapchain && resource < graph.producers.size() &&
                graph.producers[resource].size() > 1) {
                auto const writer = graph.producers[resource].back();
                if (writer >= 0 && live.contains(static_cast<std::uint32_t>(writer))) {
                    presented.push_back(writer);
                }
            }
        }
        if (!presented.empty()) {
            text += "    present [shape=doublecircle, style=\"filled\", fillcolor=\"#1A73E8\", "
                    "fontcolor=white, color=\"#1A73E8\", label=\"present\", fontsize=10];\n";
        }
        text += "\n";

        for (auto const &[key, resources]: carried) {
            auto names = std::vector<std::string>{};
            for (auto const resource: resources) {
                names.push_back(escape(graph.resources[resource].name, false));
            }
            std::ranges::sort(names);
            auto label = std::string{};
            for (auto const &name: names) {
                label += label.empty() ? "" : "\\n";
                label += name;
            }
            auto const kind = graph.resources[*resources.begin()].kind;
            auto const cross = crossing.contains(key);
            auto tooltip = std::string{};
            if (transients != nullptr) {
                for (auto const resource: resources) {
                    if (auto const *placement = transients->placement_of(resource)) {
                        tooltip += std::format("{}{}: block {} offset {} size {:.2f} MiB", tooltip.empty() ? "" : "\\n",
                                               escape(graph.resources[resource].name, false), placement->block,
                                               placement->offset, mebibytes(placement->size));
                    }
                }
            }
            text += std::format("    {} -> {} [label=\"{}\", color=\"{}\", fontcolor=\"{}\"{}{}{}];\n", key.first,
                                key.second, label, cross ? "#D93025" : kind_color(kind), kind_color(kind),
                                cross ? ", style=dashed, penwidth=1.6" : "",
                                kind == ResourceKind::token && !cross ? ", style=dotted" : "",
                                tooltip.empty() ? "" : std::format(", tooltip=\"{}\"", tooltip));
        }
        for (auto const writer: presented) {
            text += std::format("    p{} -> present [color=\"#1A73E8\", penwidth=1.6];\n", writer);
        }

        // Timeline waits between batches, drawn between the clusters.
        for (auto index = std::size_t{0}; index < compiled.batches.size(); ++index) {
            for (auto const &wait: compiled.batches[index].waits) {
                for (auto source = std::size_t{0}; source < index; ++source) {
                    auto const &other = compiled.batches[source];
                    if (other.queue != wait.queue || other.signal_index != wait.signal_index ||
                        last_pass[source] < 0 || first_pass[index] < 0) {
                        continue;
                    }
                    text += std::format("    p{} -> p{} [ltail=cluster_batch_{}, lhead=cluster_batch_{}, "
                                        "style=dotted, color=\"#9334E6\", fontcolor=\"#9334E6\", "
                                        "label=\"wait {} &ge; {}\", constraint=false];\n",
                                        last_pass[source], first_pass[index], source, index, queue_name(wait.queue),
                                        wait.signal_index);
                    break;
                }
            }
        }

        auto summary = std::format("{} passes in {} batches", passes, compiled.batches.size());
        if (!compiled.transfers.empty()) {
            summary += std::format(" \xC2\xB7 {} queue ownership transfers", compiled.transfers.size());
        }
        if (transients != nullptr) {
            summary += std::format(" \xC2\xB7 transients {:.1f} MiB aliased ({:.1f} MiB without)",
                                   mebibytes(transients->total_bytes), mebibytes(transients->unaliased_bytes));
        }
        text += std::format(
                "\n    label=<<b>Frame graph</b>  {}<br align=\"left\"/><font point-size=\"9\" color=\"#5F6368\">"
                "Solid: dependency (colour by resource kind: image, buffer, token)  \xC2\xB7  "
                "<font color=\"#D93025\">dashed red</font>: crosses queues  \xC2\xB7  "
                "<font color=\"#9334E6\">dotted purple</font>: timeline wait</font><br align=\"left\"/>>;\n",
                escape(summary, true));
        text += "}\n";
        return text;
    }

} // namespace frame_graph
