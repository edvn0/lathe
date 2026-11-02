# GPU crash dumps (Nsight Aftermath)

When the GPU faults or hangs (a shader reading memory it must not, a runaway loop), `VK_ERROR_DEVICE_LOST` says only that it
happened. The NVIDIA Nsight Aftermath SDK has the driver write a crash dump with the rest: which shader was running, the page
fault (read or write, the address), the warps that faulted, and the checkpoint each queue had reached. The engine collects it
by default where the SDK is built in.

## What you get

On a device loss the log says what the dump holds, for example:

```
[E] GPU crash dump written to ~/.local/state/lathe/gpu_crashes/lathe-20261101-134823-0.nv-gpudmp (2260928 bytes)
[E] GPU crash: page fault (Read access): Failed to translate the virtual address., address 3951960064
[E] GPU crash: active shader compute_02 is 'gi_probe_trace'
[E] GPU crash: MMU Fault Error in compute_02 @ 0x00003d90 (gi_probe_trace)
[E] GPU crash dump decoded to ~/.local/state/lathe/gpu_crashes/lathe-20261101-134823-0.json
[E] Checkpoints, graphics queue: last reached 'Probe trace' (bottom of pipe)
```

`gpu_crashes/` (in the user state directory, `$XDG_STATE_HOME/lathe`) holds the `.nv-gpudmp` (open it in Nsight Graphics, it
needs nothing else), the same dump decoded to `.json`, and in the full mode the `shader-*.nvdbg` shader debug information
Nsight Graphics looks up by the identifier in the dump. The shader names come from the engine: every shader object registers
the hash of its SPIR-V with its debug name (`gpu_crash_dumps::register_shader`). The engine's own checkpoints (the pass names of
`device_checkpoints`) are in the dump too, as text.

## Modes

`[diagnostics]` in `lathe.project` / `game.toml`:

```toml
[diagnostics]
gpu_crash_dumps = "auto"   # auto, standard, full or off
```

| mode | what it turns on | cost |
|---|---|---|
| `standard` | the crash dump and the driver's resource tracking | low enough to leave on |
| `full` | also shader debug information (line tables), automatic checkpoints with call stacks, and extra shader error reporting | CPU time in command recording and shader compilation; shaders are built from SPIR-V instead of the binary cache so the driver can add debug information |
| `off` | nothing | none |
| `auto` (default) | `standard`, and `full` for the run after a device-lost restart | |

NVIDIA's guidelines ask that the heavy options are only on in an application-controlled diagnostics mode; the restart after a
device loss is that mode here, so a crash that comes back leaves everything the driver can say. `NV_AFTERMATH_FORCE_OFF=1` in
the environment switches the library off whatever the setting says.

In `full` mode Slang compiles with standard debug information (line tables) and skips the baked shader pack where Slang is
available, so a dump maps to shader source lines.

## Building with it

The SDK is not in the repository (NVIDIA's licence does not allow vendoring it). Download the Linux package from
<https://developer.nvidia.com/nsight-aftermath>, extract it, and point `AFTERMATH_SDK_DIR` at the directory with `include/`
and `lib/x64/` (default: `~/builds/nsight-aftermath-sdk`); `cargo xtask configure` passes it on and mounts it into the build
container. Without it the build uses `src/gpu/aftermath/stub.cxx` and nothing is collected. The library itself is opened at
run time (`dlopen`), so a build that has the SDK still starts on a machine that does not: it just collects nothing. Needs the
NVIDIA driver 455 or newer on Linux (515 for the extra shader error reporting) and the extensions
`VK_NV_device_diagnostic_checkpoints` and `VK_NV_device_diagnostics_config`.

## Testing it

There is no command that crashes the GPU on purpose, since `game.debug_lose_device()` only fakes the API error. To check the
whole path, temporarily make a shader read outside its buffer (the emitter grid header is a convenient place) and run a scene:
the dump, its decoding and the log lines above appear, and with a device loss the engine restarts in `full`.
