#pragma once

#include <cstdint>

#include "core/handle.hxx"

// Full definitions live in model_storage.hxx/mesh_storage.hxx, alongside the
// ObjectPool<..., 0> instances they back (see sampler.hxx's SamplerHandle
// for why an incomplete forward declaration is enough here).
//
// Sentinel = 0: index 0 is permanently wasted (never allocated -- see
// ModelStorage::create()/MeshStorage::create()) purely so a
// default-constructed handle's index can never collide with a real slot,
// matching MaterialHandle's convention.
struct ModelSlotData;
struct MeshSlotData;

using ModelHandle = Handle<ModelSlotData, 0>;
using MeshHandle = Handle<MeshSlotData, 0>;
