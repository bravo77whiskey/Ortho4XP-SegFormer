"""Real photographs behind the photo texture set, and how each is cut.

Every source is public domain or CC0, checked again by ``fetch_sources.py``
against the file page's license field before it is used:

* USDA photos on Wikimedia Commons (U.S. federal government work, public
  domain): polytunnel film and open tunnel ends;
* CC0 door photos on Wikimedia Commons;
* Poly Haven photoscans (polyhaven.com/license: CC0) for brick, render,
  thatch, fabric, soil and concrete.

No source is a photo of a Chinese greenhouse: Commons and Openverse have
no public-domain or CC0 pictures of solar greenhouses, their quilts or a
polytunnel end wall with a door.  The tunnels here are Californian
raspberry tunnels (same film-over-hoops build); the solar greenhouse's
quilts, straw and walls are material scans of the same materials.

Pixel coordinates refer to the Commons rendition ``fetch_sources.py``
downloads (``COMMONS_WIDTH`` wide).
"""

COMMONS_WIDTH = 3840

USDA_0050 = "File:Commercial High Tunnel (20181115-FPAC-LSC-0050).jpg"
USDA_0048 = "File:Commercial High Tunnel (20181115-FPAC-LSC-0048).jpg"
USDA_0043 = "File:Commercial High Tunnel (20181115-FPAC-LSC-0043).jpg"
USDA_0038 = "File:Commercial High Tunnel (20181115-FPAC-LSC-0038).jpg"
DOOR_RUSTY = "File:Porte en métal rouillé, Schulerstraße (Vienne), août 2019.jpg"
DOOR_BLUE = "File:Madrid - Puerta azul en el barrio de Valdeacederas.jpg"

COMMONS = (USDA_0050, USDA_0048, USDA_0043, USDA_0038, DOOR_RUSTY, DOOR_BLUE)

# Poly Haven id -> resolution of the colour map
POLYHAVEN = {
    "worn_brick_wall": "2k",       # back wall + gable (variant a), anteroom
    "concrete_wall_008": "2k",     # back wall + gable, variant b (render)
    "dirty_concrete": "2k",        # anteroom roof
    "thatch_roof_angled": "2k",    # straw back roof, rolled straw mats
    "knitted_fleece": "1k",        # rolled quilts
    "denim_fabric_05": "1k",       # rolled quilts
    "brown_mud_dry": "2k",         # soil at the film feet
}

# ---- film: one raspberry tunnel in USDA 0050, seen from its end and above.
# The two dark gaps beside it (rough lines, refined on the photo) are its
# feet; the film between them is unrolled from y FILM_Y[0] (far) to
# FILM_Y[1] (near).  The left flank is seen nearly edge-on and the right
# foot drops out of view, so only the middle FILM_COLUMNS of the band
# (the top of the tunnel and the flank facing the camera) are used and
# stretched over the whole arch.
FILM_SOURCE = USDA_0050
FILM_GAP_LEFT = ((-9, 236), (798, 1380))
FILM_GAP_RIGHT = ((685, 162), (1750, 1242))
FILM_Y = (200, 1500)
FILM_COLUMNS = (0.32, 0.88)        # of the unrolled band, across
# hoop windows (index of the first hoop, counted from the near end) per use
FILM_WINDOWS = {"roof_a": 0, "solar": 3, "roof_b": 7}

# ---- tunnel gables: open tunnel ends seen nearly straight on.  Quad =
# (ground-left, ground-right, apex); the top edge runs through the apex
# parallel to the ground line.
GABLE_ENDS = {
    "gable_0": (USDA_0048, (700, 4900), (3480, 4680), (2024, 3302)),
    "gable_1": (USDA_0043, (930, 2080), (3020, 2090), (1976, 1008)),
    "gable_2": (USDA_0038, (1610, 1600), (2420, 1540), (1976, 988)),
    "gable_3": (USDA_0038, (2420, 1540), (3130, 1420), (2738, 893)),
}

# ---- doors pasted onto the masonry: (x0, y0, x1, y1) of the leaf and its
# frame in the photo.  The rusty door is the right-hand pair of panels of a
# double door (the left pair carries a painted house name); the blue door
# is the left leaf of a double door, below its transom.
DOORS = {
    "rusty": (DOOR_RUSTY, (918, 237, 1566, 1779)),
    "blue": (DOOR_BLUE, (330, 750, 765, 1810)),
}
# gable door (solar greenhouses without an anteroom): width, height in
# metres and its distance from the north end of the gable
GABLE_DOOR_M = (0.85, 1.95, 0.9)
