# Environments

HDR cubemaps for image-based lighting and the skybox.

| File | Source | License |
| --- | --- | --- |
| `belfast_sunset_puresky_512.ktx2` | [Belfast Sunset (Pure Sky)](https://polyhaven.com/a/belfast_sunset_puresky) from Poly Haven. Photography by Dimitrios Savva, processing by Greg Zaal, sky edits by Jarod Guest. | CC0 1.0 |

CC0 needs no attribution; the credits are kept as a courtesy.

## Regenerating

The `.ktx2` is the 4K equirect `.hdr` projected to a cube by `lathe-env-bake` (E5B9G9R9, Zstd supercompressed, 512
faces, 4x4 supersampling):

```
curl -LO https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/belfast_sunset_puresky_4k.hdr
cargo xtask build
build/debug/bin/lathe-env-bake belfast_sunset_puresky_4k.hdr assets/environments/belfast_sunset_puresky_512.ktx2 \
    --size 512 --preview cross.png
```

`--preview` writes a tonemapped cross of the six faces, which is the quickest way to check orientation and seams.
