#pragma once

#include "core/handle.hxx"

// Defined in rendering/script_storage.hxx; only the handle lives here so the scene layer doesn't depend on
// IScript.
//
// Sentinel = 0: slot 0 is reserved, so a default ScriptHandle means "no script".
struct ScriptSlotData;
using ScriptHandle = Handle<ScriptSlotData, 0>;
