
// ASSUMPTION: stb_image for CPU-side decode of encoded (PNG/JPEG) texture
// bytes into RGBA8 pixels. If you already have a decode path elsewhere,
// delete this include and swap decode_image()'s body to call it instead.
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
