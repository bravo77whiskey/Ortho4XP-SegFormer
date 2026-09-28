"""Atlas layout shared by the geometry and both texture sets.

Two 1024 x 1024 pages, ``tunnels`` and ``solar``.  Each region is a pixel
box ``(x0, y0, x1, y1)`` with row 0 at the top of the image.  A region's
texture is authored in its natural orientation: ``across`` runs left to
right in the source image and ``along`` runs bottom to top.  Rotated
regions store the source turned a quarter turn clockwise, so a long thin
subject fits a tall thin box.

Regions whose ``along`` axis spans the full page height tile along it (the
page wraps in v), so geometry maps ``along`` metres / ``period_m`` straight
into v without splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

PAGE_PX = 1024


@dataclass(frozen=True)
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    rotated: bool = False
    period_m: float | None = None   # along-axis tiling period (full height)

    @property
    def size_px(self):
        """(across, along) pixel size of the source image."""
        w, h = self.x1 - self.x0, self.y1 - self.y0
        return (h, w) if self.rotated else (w, h)

    def uv(self, a, b):
        """Texture (u, v) for ``a`` across (0-1) and ``b`` along (0-1, or
        metres / period for tiling regions)."""
        if not self.rotated:
            u = (self.x0 + a * (self.x1 - self.x0)) / PAGE_PX
            v = 1.0 - (self.y1 - b * (self.y1 - self.y0)) / PAGE_PX
            return u, v
        # source turned clockwise: source across -> page top-to-bottom,
        # source along (bottom to top) -> page left-to-right
        u = (self.x0 + b * (self.x1 - self.x0)) / PAGE_PX
        v = 1.0 - (self.y0 + a * (self.y1 - self.y0)) / PAGE_PX
        return u, v


# one source image per repeat: the film images show about 12 hoops
ALONG_PERIOD_M = 12.0

TUNNELS = {
    "roof_a": Region(0, 0, 384, 1024, period_m=ALONG_PERIOD_M),
    "roof_b": Region(384, 0, 768, 1024, period_m=ALONG_PERIOD_M),
    "gable_0": Region(768, 0, 1024, 256),
    "gable_1": Region(768, 256, 1024, 512),
    "gable_2": Region(768, 512, 1024, 768),
    "gable_3": Region(768, 768, 1024, 1024),
}
TUNNEL_GABLES = ("gable_0", "gable_1", "gable_2", "gable_3")
# arch fraction at each foot covered by soil (berms map here)
FOOT_SOIL = 0.03
# door in the tunnel gable image (across = width, along = height), as
# (a0, a1, b0, b1) fractions
TUNNEL_DOOR = (0.42, 0.58, 0.0, 0.62)
# per gable image; a texture set may override (its pages' doors.json)
TUNNEL_DOORS = {name: TUNNEL_DOOR for name in TUNNEL_GABLES}

SOLAR = {
    "film": Region(0, 0, 448, 1024, period_m=ALONG_PERIOD_M),
    "quilt": Region(448, 0, 544, 1024, period_m=ALONG_PERIOD_M),
    "back": Region(544, 0, 640, 1024, period_m=ALONG_PERIOD_M),
    "wall_a": Region(640, 0, 768, 1024, period_m=ALONG_PERIOD_M),
    "wall_b": Region(768, 0, 896, 1024, period_m=ALONG_PERIOD_M),
    "gable_a": Region(896, 0, 1024, 256, rotated=True),
    "gable_b": Region(896, 256, 1024, 512, rotated=True),
    "ante_wall": Region(896, 512, 1024, 832, rotated=True),
    "ante_roof": Region(896, 832, 1024, 1024),
}
# back wall and gable finish per solar variant (brick / cement render)
SOLAR_FINISH = {"a": ("wall_a", "gable_a"), "b": ("wall_b", "gable_b")}
# Solar gable image: across = width (d, south foot to back of wall),
# along = height.  Anteroom wall image: across = perimeter distance (south
# face, east face, north face), along = height; the door is on the south
# face at (a0, a1, b0, b1).
ANTE_DOOR = (0.10, 0.17, 0.0, 0.78)
