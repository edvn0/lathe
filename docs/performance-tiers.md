# Performance tiers

The engine classes the GPU as `low`, `medium` or `high` at startup (`gpu/performance_tier.hxx`) and logs it
(`Performance tier: ...`). Settings left on `auto` follow the tier; setting one explicitly overrides it.

- **Detection.** A discrete or virtual GPU (an external GPU is discrete) is `high` with 6 GiB or more of device local
  memory and `medium` below; integrated, CPU and unknown devices are `low`.
- **Override.** `[render] performance_tier = "auto" | "low" | "medium" | "high"` (Edit > Project Settings > Rendering,
  applied the next time the project opens).

| Setting | low | medium | high |
| --- | --- | --- | --- |
| `[render] ray_traced_shadow_resolution = "auto"` | quarter | half | full |
| `[gi] far_update_interval = 0` (auto) | every 4 frames | every 2 | every 2 |

A new setting joins the ladder by adding a field to `TierProfile` and an `auto` value that reads it.
