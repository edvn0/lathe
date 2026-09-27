#include "rendering/overlay.hxx"

#include <algorithm>
#include <bit>
#include <tuple>

OverlayRegistration::~OverlayRegistration() { reset(); }

OverlayRegistration::OverlayRegistration(OverlayRegistration &&other) noexcept :
    registry_{std::exchange(other.registry_, nullptr)}, id_{std::exchange(other.id_, 0)} {}

auto OverlayRegistration::operator=(OverlayRegistration &&other) noexcept -> OverlayRegistration & {
    if (this != &other) {
        reset();
        registry_ = std::exchange(other.registry_, nullptr);
        id_ = std::exchange(other.id_, 0);
    }

    return *this;
}

auto OverlayRegistration::reset() noexcept -> void {
    if (registry_ != nullptr) {
        registry_->remove(id_);
        registry_ = nullptr;
        id_ = 0;
    }
}

OverlayRegistry::IterationGuard::IterationGuard(OverlayRegistry &registry) noexcept : registry_{registry} {
    ++registry_.iteration_depth_;
}

OverlayRegistry::IterationGuard::~IterationGuard() {
    if (--registry_.iteration_depth_ == 0) {
        registry_.flush_pending();
    }
}

OverlayRegistry::OverlayRegistry() {
    // remove() is noexcept (it runs from OverlayRegistration's destructor)
    // and may queue while the renderer iterates; never let that allocate.
    pending_removals_.reserve(max_overlays);
}

auto OverlayRegistry::add(OverlayDesc desc) -> std::expected<OverlayRegistration, OverlayRegistryError> {
    if (desc.name.empty()) {
        return std::unexpected(OverlayRegistryError::empty_name);
    }

    if (!desc.record) {
        return std::unexpected(OverlayRegistryError::missing_record_callback);
    }

    if (desc.stage >= OverlayStage::count) {
        return std::unexpected(OverlayRegistryError::invalid_stage);
    }

    if (std::popcount(used_slots_) >= static_cast<int>(max_overlays)) {
        return std::unexpected(OverlayRegistryError::capacity_exceeded);
    }

    auto const slot = static_cast<std::uint32_t>(std::countr_one(used_slots_));
    used_slots_ |= 1U << slot;

    Entry entry{
            .id = next_id_++,
            .slot = slot,
            .sequence = next_sequence_++,
            .desc = std::move(desc),
    };

    auto const id = entry.id;

    if (iteration_depth_ > 0) {
        pending_additions_.push_back(std::move(entry));
    } else {
        insert(std::move(entry));
    }

    return OverlayRegistration{*this, id};
}

auto OverlayRegistry::remove(OverlayId id) noexcept -> void {
    // A registration that never left the pending queue can go right away;
    // nothing is iterating over pending_additions_.
    if (auto const pending = std::ranges::find(pending_additions_, id, &Entry::id);
        pending != pending_additions_.end()) {
        used_slots_ &= ~(1U << pending->slot);
        pending_additions_.erase(pending);
        return;
    }

    if (iteration_depth_ > 0) {
        pending_removals_.push_back(id);
        return;
    }

    erase(id);
}

auto OverlayRegistry::stage(OverlayStage stage) noexcept -> std::span<Entry> {
    auto const [first, last] =
            std::ranges::equal_range(entries_, stage, {}, [](Entry const &entry) { return entry.desc.stage; });

    return {first, last};
}

auto OverlayRegistry::insert(Entry entry) -> void {
    auto const key = [](Entry const &e) { return std::tuple{e.desc.stage, e.desc.order, e.sequence}; };

    auto const position = std::ranges::upper_bound(entries_, key(entry), {}, key);
    entries_.insert(position, std::move(entry));
}

auto OverlayRegistry::erase(OverlayId id) noexcept -> void {
    if (auto const found = std::ranges::find(entries_, id, &Entry::id); found != entries_.end()) {
        used_slots_ &= ~(1U << found->slot);
        entries_.erase(found);
    }
}

auto OverlayRegistry::flush_pending() -> void {
    // clear() rather than swapping in a fresh vector: keep the capacity the
    // constructor reserved, so remove() never allocates.
    for (auto const id: pending_removals_) {
        erase(id);
    }

    pending_removals_.clear();

    for (auto &entry: std::exchange(pending_additions_, {})) {
        insert(std::move(entry));
    }
}
