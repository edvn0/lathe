#pragma once

#include <cstdint>

#include "core/handle.hxx"

struct ModelSlotData;
struct MeshSlotData;

using ModelHandle = Handle<ModelSlotData, 0>;
using MeshHandle = Handle<MeshSlotData, 0>;
