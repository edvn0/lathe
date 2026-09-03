#pragma once

#include "rendering/entity.hxx"

class IScript {
public:
    virtual ~IScript() = default;

    virtual auto on_attach(Entity) -> void {}
    virtual auto on_detach(Entity) -> void {}

    virtual auto on_update(ScriptEntity, float) -> void {}

    [[nodiscard]] virtual auto parallelizable() const noexcept -> bool { return false; }
};
