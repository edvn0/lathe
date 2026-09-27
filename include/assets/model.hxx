#pragma once

#include <cstdint>

#include "core/handle.hxx"

// Defined in model_storage.hxx and mesh_storage.hxx.
//
// Sentinel = 0: index 0 is never allocated, so a default handle never names a real slot.
struct ModelSlotData;
struct MeshSlotData;

using ModelHandle = Handle<ModelSlotData, 0>;
using MeshHandle = Handle<MeshSlotData, 0>;
