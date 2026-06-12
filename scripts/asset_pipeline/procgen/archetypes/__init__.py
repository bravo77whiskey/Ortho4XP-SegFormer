"""Registry of implemented procedural building archetypes.

Each builder is ``build(length_m, width_m, floors, seed, layout) -> MeshSpec``
where ``layout`` is the parsed atlas_layout.json.  Builders are pure Python
(no bpy) and must return a spec that passes ``common.validate_spec`` for the
requested footprint.

grid.py marks manifest rows for archetypes missing from this registry as
``enabled: false`` so the library can be regenerated incrementally while
archetypes land one by one.
"""

from .apartment import build_aptblock, build_aptslab
from .commercial import build_flatcom
from .flatroof import build_flatres, build_shophouse
from .house import build_gable, build_hip, build_lshape
from .industrial import build_bigbox, build_warehouse
from .rowhouse import build_rowhouse

ARCHETYPES = {
    "gable": build_gable,
    "hip": build_hip,
    "lshape": build_lshape,
    "rowhouse": build_rowhouse,
    "flatres": build_flatres,
    "shophouse": build_shophouse,
    "aptslab": build_aptslab,
    "aptblock": build_aptblock,
    "flatcom": build_flatcom,
    "warehouse": build_warehouse,
    "bigbox": build_bigbox,
}


def build_archetype(name, length_m, width_m, floors, seed, layout,
                    flavor="generic"):
    try:
        builder = ARCHETYPES[name]
    except KeyError:
        raise KeyError(
            f"archetype {name!r} not implemented; have {sorted(ARCHETYPES)}"
        ) from None
    return builder(length_m, width_m, floors, seed, layout, flavor)
