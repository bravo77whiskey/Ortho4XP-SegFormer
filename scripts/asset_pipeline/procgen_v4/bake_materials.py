"""Blender node trees for the bake scenes (runs inside Blender only).

Every material is a Diffuse BSDF: the bake wants albedo x occlusion under a
uniform white sky, never specular highlights.  Gloss, metalness and the
weathering masks leave through Shader AOVs instead.

Noise is sampled on a cylinder (x -> angle with circumference W) so every
procedural pattern repeats exactly with the strip's world width and the
atlas strip wraps in u without a seam.
"""

import math

import bpy


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_rgba(value):
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b), 1.0)


class _T:
    """Tiny node-tree helper."""

    def __init__(self, material):
        self.nt = material.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def n(self, kind, **props):
        node = self.nt.nodes.new(kind)
        node.location = (self.x, 0)
        self.x += 180
        for key, value in props.items():
            setattr(node, key, value)
        return node

    def link(self, a, b):
        self.nt.links.new(a, b)

    def val(self, v):
        node = self.n("ShaderNodeValue")
        node.outputs[0].default_value = float(v)
        return node.outputs[0]

    def math(self, op, a, b=None, clamp=False):
        node = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                node.inputs[i].default_value = float(v)
            else:
                self.link(v, node.inputs[i])
        return node.outputs[0]

    def mix(self, a, b, fac, blend="MIX"):
        node = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        for sock, v in ((node.inputs[6], a), (node.inputs[7], b),
                        (node.inputs[0], fac)):
            if isinstance(v, (int, float)):
                sock.default_value = float(v)
            elif isinstance(v, tuple):
                sock.default_value = v
            else:
                self.link(v, sock)
        return node.outputs[2]


def _attr(t, name):
    node = t.n("ShaderNodeAttribute", attribute_type="GEOMETRY",
               attribute_name=name)
    return node.outputs["Fac"]


def _periodic_coords(t, period):
    """(R cos(2 pi x / W), R sin(...), z) with R = W / 2 pi."""
    geo = t.n("ShaderNodeNewGeometry")
    sep = t.n("ShaderNodeSeparateXYZ")
    t.link(geo.outputs["Position"], sep.inputs[0])
    ang = t.math("MULTIPLY", sep.outputs[0], 2 * math.pi / period)
    radius = period / (2 * math.pi)
    cx = t.math("MULTIPLY", t.math("COSINE", ang), radius)
    cy = t.math("MULTIPLY", t.math("SINE", ang), radius)
    comb = t.n("ShaderNodeCombineXYZ")
    t.link(cx, comb.inputs[0])
    t.link(cy, comb.inputs[1])
    t.link(sep.outputs[2], comb.inputs[2])
    return comb.outputs[0], sep.outputs[0], sep.outputs[2], sep.outputs[1]


def _noise(t, vec, scale, detail=4.0, rough=0.55, w=None):
    node = t.n("ShaderNodeTexNoise", noise_dimensions="4D" if w else "3D")
    node.inputs["Scale"].default_value = scale
    node.inputs["Detail"].default_value = detail
    node.inputs["Roughness"].default_value = rough
    if w:
        node.inputs["W"].default_value = w
    t.link(vec, node.inputs["Vector"])
    return node.outputs["Fac"]


def _aov(t, name, value):
    node = t.n("ShaderNodeOutputAOV")
    node.aov_name = name
    if isinstance(value, (int, float)):
        node.inputs["Value"].default_value = float(value)
    else:
        t.link(value, node.inputs["Value"])


def _variation(t, value, amount):
    """1 + amount * (2 value - 1)."""
    return t.math("ADD", t.math("MULTIPLY", t.math("SUBTRACT", value, 0.5),
                                2.0 * amount), 1.0)


def _scale_color(t, color, factor):
    node = t.n("ShaderNodeVectorMath", operation="SCALE")
    t.link(color, node.inputs[0])
    if isinstance(factor, (int, float)):
        node.inputs["Scale"].default_value = factor
    else:
        t.link(factor, node.inputs["Scale"])
    return node.outputs[0]


def build_material(name, spec, period, seed=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    t = _T(material)
    vec, px, pz, py = _periodic_coords(t, period)
    rnd = _attr(t, "rnd")
    rnd2 = _attr(t, "rnd2")

    # ---- base colour from the palette
    palette = spec["base"]
    ramp = t.n("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = (
        "CONSTANT" if spec.get("mode", "rnd") == "rnd" else "LINEAR")
    elems = ramp.color_ramp.elements
    while len(elems) > 1:
        elems.remove(elems[-1])
    k = len(palette)
    for i, colour in enumerate(palette):
        pos = (i / k) if spec.get("mode", "rnd") == "rnd" else (
            i / max(1, k - 1))
        el = elems[0] if i == 0 else elems.new(pos)
        el.position = pos
        el.color = hex_rgba(colour)
    t.link(rnd, ramp.inputs[0])
    color = ramp.outputs[0]

    # per-piece value jitter (tiles, bricks, boards)
    pv = spec.get("piece_var", 0.0)
    if pv:
        color = _scale_color(t, color, _variation(t, rnd2, pv))

    # large + fine periodic noise
    nscale, namount = spec.get("noise", (0.6, 0.06))
    fscale, famount = spec.get("fine", (9.0, 0.035))
    if namount:
        color = _scale_color(t, color, _variation(
            t, _noise(t, vec, nscale, 3.0, 0.5, w=1.0 + seed), namount))
    if famount:
        color = _scale_color(t, color, _variation(
            t, _noise(t, vec, fscale, 6.0, 0.65, w=7.0 + seed), famount))
    # a hue drift between two palette ends (e.g. faded vs fresh paint)
    drift = spec.get("drift")
    if drift:
        dn = _noise(t, vec, drift[1], 2.0, 0.4, w=3.0 + seed)
        dn = t.math("MULTIPLY", t.math("SUBTRACT", dn, 0.35), drift[2],
                    clamp=True)
        color = t.mix(color, hex_rgba(drift[0]), dn)

    height = None
    pat = spec.get("pattern") or {}
    ptype = pat.get("type")
    if ptype in ("brick", "tiles", "panels", "block"):
        brick = t.n("ShaderNodeTexBrick")
        brick.offset = pat.get("offset", 0.5 if ptype != "tiles" else 0.0)
        brick.offset_frequency = 2
        brick.squash = 1.0
        brick.inputs["Scale"].default_value = 1.0
        brick.inputs["Mortar Size"].default_value = pat.get("mortar", 0.01)
        brick.inputs["Mortar Smooth"].default_value = pat.get("smooth", 0.1)
        brick.inputs["Bias"].default_value = 0.0
        brick.inputs["Brick Width"].default_value = pat["w"]
        brick.inputs["Row Height"].default_value = pat["h"]
        brick.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1)
        brick.inputs["Color2"].default_value = (1.2, 1.2, 1.2, 1)
        brick.inputs["Mortar"].default_value = (0, 0, 0, 1)
        comb = t.n("ShaderNodeCombineXYZ")
        t.link(px, comb.inputs[0])
        t.link(pz, comb.inputs[1])
        t.link(comb.outputs[0], brick.inputs["Vector"])
        # brick colour variation (per brick) multiplies the base
        bw = t.n("ShaderNodeRGBToBW")
        t.link(brick.outputs["Color"], bw.inputs[0])
        bvar = pat.get("var", 0.18)
        color = _scale_color(t, color, t.math(
            "ADD", t.math("MULTIPLY", t.math("SUBTRACT", bw.outputs[0], 1.0),
                          bvar / 0.2), 1.0))
        if pat.get("alt"):
            # a second brick population (burnt headers, mixed stock)
            sel = t.math("GREATER_THAN", bw.outputs[0], pat.get("alt_th", 1.12))
            color = t.mix(color, hex_rgba(pat["alt"]), sel)
        mortar = brick.outputs["Fac"]
        color = t.mix(color, hex_rgba(pat.get("mortar_color", "#8a8580")),
                      mortar)
        height = t.math("SUBTRACT", 1.0, mortar)
    elif ptype in ("siding", "slats", "boards"):
        hh = pat["h"]
        f = t.math("FRACT", t.math("DIVIDE", pz, hh))
        if ptype == "boards":
            # vertical boards: use x
            f = t.math("FRACT", t.math("DIVIDE", px, hh))
        # lap profile: thick at the bottom of each board
        height = t.math("SUBTRACT", 1.0, f)
        shade = t.math("ADD", t.math("MULTIPLY", t.math(
            "POWER", f, 6.0), -pat.get("shadow", 0.25)), 1.0)
        shade2 = t.math("ADD", t.math("MULTIPLY", t.math(
            "LESS_THAN", f, 0.06), -pat.get("shadow", 0.25) * 0.8), 1.0)
        color = _scale_color(t, _scale_color(t, color, shade), shade2)
    elif ptype == "granules":
        g = _noise(t, vec, pat.get("scale", 160.0), 2.0, 0.5, w=11.0 + seed)
        color = _scale_color(t, color, _variation(t, g, pat.get("amount",
                                                                0.25)))
        height = g
    elif ptype == "grain":
        wave = t.n("ShaderNodeTexWave", wave_type="BANDS",
                   bands_direction="Z")
        wave.inputs["Scale"].default_value = pat.get("scale", 6.0)
        wave.inputs["Distortion"].default_value = 6.0
        wave.inputs["Detail"].default_value = 3.0
        t.link(vec, wave.inputs["Vector"])
        color = _scale_color(t, color, _variation(t, wave.outputs["Fac"],
                                                  pat.get("amount", 0.12)))

    # ---- bump: pattern height plus a fine surface noise
    bump_amt = spec.get("bump", 0.0)
    normal = None
    if bump_amt or height is not None:
        hsum = height
        tex = _noise(t, vec, spec.get("bump_scale", 30.0), 5.0, 0.6,
                     w=5.0 + seed)
        if hsum is None:
            hsum = tex
        else:
            hsum = t.math("ADD", hsum, t.math("MULTIPLY", tex, 0.15))
        bump = t.n("ShaderNodeBump")
        bump.inputs["Strength"].default_value = max(0.05, bump_amt or 0.4)
        bump.inputs["Distance"].default_value = spec.get("bump_dist", 0.004)
        t.link(hsum, bump.inputs["Height"])
        normal = bump.outputs["Normal"]

    diffuse = t.n("ShaderNodeBsdfDiffuse")
    t.link(color, diffuse.inputs["Color"])
    if normal is not None:
        t.link(normal, diffuse.inputs["Normal"])
    shader = diffuse.outputs[0]
    if spec.get("emit"):
        em = t.n("ShaderNodeEmission")
        em.inputs["Color"].default_value = hex_rgba(spec["emit"])
        em.inputs["Strength"].default_value = 1.0
        add = t.n("ShaderNodeAddShader")
        t.link(shader, add.inputs[0])
        t.link(em.outputs[0], add.inputs[1])
        shader = add.outputs[0]
    out = t.n("ShaderNodeOutputMaterial")
    t.link(shader, out.inputs["Surface"])

    rough = spec.get("rough", 0.85)
    gloss_var = spec.get("gloss_var", 0.0)
    gloss = 1.0 - rough
    if gloss_var:
        gloss = t.math("ADD", t.math("MULTIPLY", t.math(
            "SUBTRACT", rnd2, 0.5), gloss_var), gloss, clamp=True)
    _aov(t, "gloss", gloss)
    _aov(t, "metal", spec.get("metal", 0.0))
    _aov(t, "drip", spec.get("drip", 0.0))
    _aov(t, "cls", spec.get("cls", 1) / 16.0)
    _aov(t, "rnd", rnd)
    _aov(t, "depth", py)
    return material
