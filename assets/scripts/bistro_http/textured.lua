-- Bistro from a textured copy on disk (NVIDIA ORCA's BistroExterior.glb with its Textures folder next to it; see scene.lua).
-- Started with --game=lua --script=assets/scripts/bistro_http/textured.lua --player [--max-quality].
local scene = require("bistro_http.scene")
scene.configure({ local_model = "/home/edwin/Downloads/Bistro_v5_2/BistroExterior.glb" })
return scene
