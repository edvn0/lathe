-- What the menu offers. The first entry of each list is the default.

local Config = {}

-- Cells per side. All of these boards are built once when the game starts; choosing one only moves the camera.
Config.sizes = {
    { name = "Small", width = 12, height = 12 },
    { name = "Medium", width = 20, height = 20 },
    { name = "Large", width = 32, height = 24 },
}

-- Snake speed in cells per second.
Config.speeds = {
    { name = "Slow", value = 5 },
    { name = "Normal", value = 9 },
    { name = "Fast", value = 15 },
}

-- How often food appears and how much can lie around at once.
Config.spawns = {
    { name = "Sparse", interval = 0, max_food = 1 },
    { name = "Steady", interval = 4, max_food = 3 },
    { name = "Frenzy", interval = 1.5, max_food = 8 },
}

return Config
