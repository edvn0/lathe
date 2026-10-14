return {
    properties = {
        { name = "label", type = "string", default = "none" },
        { name = "target", type = "entity" },
        { name = "tex", type = "texture" },
        { name = "mat", type = "material" },
        { name = "count", type = "integer", default = 2, min = 0, max = 9 },
    },
    on_start = function(self)
        print(("probe: label=%s target=%s tex=%s mat=%s count=%d"):format(self.label, tostring(self.target and self.target:name()),
            tostring(self.tex), tostring(self.mat), self.count))
    end,
}
