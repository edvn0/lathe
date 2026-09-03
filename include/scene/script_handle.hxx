#pragma once

#include "core/handle.hxx"

// Full definition (ScriptSlotData holds the actual std::unique_ptr<IScript>)
// lives in rendering/script_storage.hxx, in the engine_rendering module --
// IScript needs the complete Entity/Scene types (see entity.hxx), which
// engine_scene must not depend on (see MODULARIZATION_HANDOFF.md's
// "Scene/Entity reclassification"). Only the opaque handle lives here so
// Components::Script can reference it without that dependency.
//
// Sentinel = 0: index 0 is permanently reserved/unused (see
// ScriptStorage::create()), matching MaterialHandle/ModelHandle's
// convention, so a default-constructed ScriptHandle unambiguously means
// "no script attached".
struct ScriptSlotData;
using ScriptHandle = Handle<ScriptSlotData, 0>;
