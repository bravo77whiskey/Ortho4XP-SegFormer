"""Ortho4XP configuration variables."""

import O4_OSM_Utils as OSM


global_prefix = "global_"

cfg_app_vars = {
    # App
    "verbosity": {
        "module": "UI",
        "type": int,
        "default": 1,
        "values": (0, 1, 2, 3),
        "hint": "Verbosity determines the amount of information about the whole process which is printed on screen.  Critical errors, if any, are reported in all states as well as in the Log. Values above 1 are probably only useful for for debug purposes.",
    },
    "cleaning_level": {
        "module": "UI",
        "type": int,
        "default": 1,
        "values": (0, 1, 2, 3),
        "hint": "Determines which temporary files are removed. Level 3 erases everything except the config and what is needed for X-Plane; Level 2 erases everything except what is needed to redo the current step only; Level 1 allows you to redo any prior step; Level 0 keeps every single file.",
    },
    "ui_theme": {
        "module": "THEME",
        "type": str,
        "default": "light",
        "values": ["light", "dark"],
        "hint": "Application color theme. Dark mode is applied when Ortho4XP is restarted or when configuration windows are reopened.",
    },
    "overpass_server_choice": {
        "module": "OSM",
        "type": str,
        "default": "random",
        "values": ["random"] + sorted(OSM.overpass_servers.keys()),
        "hint": "The (country) of the Overpass OSM server used to grab vector data. It can be modified on the fly (as all _Application_ variables) in case of problem with a particular server.",
    },
    "osm_pbf_dir": {
        "module": "PBF",
        "type": str,
        "default": "",
        "hint": "Directory holding your local planet.osm.pbf and its filtered scenery extract. When set (and the extract has been built through the Local OSM data manager window), tile builds slice OSM data locally instead of querying Overpass servers. Leave empty to keep using Overpass.",
    },
    "skip_downloads": {
        "module": "TILE",
        "type": bool,
        "default": False,
        "hint": "Will only build the DSF and TER files but not the textures (neither download nor convert). This could be useful in cases where imagery cannot be shared.",
    },
    "skip_converts": {
        "module": "TILE",
        "type": bool,
        "default": False,
        "hint": "Imagery will be downloaded but not converted from jpg to dds. Some user prefer to postprocess imagery with third party softwares prior to the dds conversion. In that case Step 3 needs to be run a second time after the retouch work.",
    },
    "max_download_slots": {
        "module": "TILE",
        "type": int,
        "default": 1,
        "values": tuple(range(1, 9)),
        "short_name": "Concurrent texture downloads",
        "hint": "Number of 4096x4096 orthophotos to download concurrently. Each orthophoto normally uses 64 network-request threads, so a value of 2 can issue up to 128 requests at once. Start with 1 or 2; higher values use more memory, may overload slower drives, and can trigger provider limits.",
    },
    "max_convert_slots": {
        "module": "TILE",
        "type": int,
        "default": 4,
        "hint": "Number of parallel threads for dds conversion. Should be mainly dictated by the number of cores in your CPU.",
    },
    "check_tms_response": {
        "module": "IMG",
        "type": bool,
        "default": True,
        "hint": "When set, internal server errors (HTTP [500] and the likes) yields new requests, if not a white texture is used in place.",
    },
    "http_timeout": {
        "module": "IMG",
        "type": float,
        "default": 10.0,
        "hint": "Delay before we decide that a http request is timed out.",
    },
    "max_connect_retries": {
        "module": "IMG",
        "type": int,
        "default": 5,
        "hint": "How much times do we try again after a failed connection for imagery request. Only used if check_tms_response is set to True.",
    },
    "max_baddata_retries": {
        "module": "IMG",
        "type": int,
        "default": 5,
        "hint": "How much times do we try again after an internal server error for an imagery request. Only used if check_tms_response is set to True.",
    },
    "ovl_exclude_pol": {
        "module": "OVL",
        "type": list,
        "default": [0],
        "hint": 'Indices of polygon types which one would like to left aside in the extraction of overlays. The list of these indices in front of their name can be obtained by running the "extract overlay" process with verbosity = 2 (skip facades that can be numerous) or 3. Index 0 corresponds to beaches in Global and HD sceneries. Strings can be used in places of indices, in that case any polygon_def that contains that string is excluded, and the string can begin with a ! to invert the matching. As an exmaple, ["!.for"] would exclude everything but forests.',
    },
    "ovl_exclude_net": {
        "module": "OVL",
        "type": list,
        "default": [],
        "hint": "Indices of road types which one would like to left aside in the extraction of overlays. The list of these indices is can be in the roads.net file within X-Plane Resources, but some sceneries use their own corresponding net definition file. Powerlines have index 22001 in XP11 roads.net default file.",
    },
    "auto_link_custom_scenery": {
        "type": bool,
        "default": True,
        "short_name": "auto_link_scenery",
        "hint": "When set, every tile whose DSF has just been built is automatically linked into custom_scenery_dir (a junction on Windows, a symlink on Mac/Linux). Existing links are matched by their target, so a tile you renamed by hand is never linked twice. Requires custom_scenery_dir to be set; has no effect otherwise.",
    },
    "custom_scenery_dir": {
        "type": str,
        "default": "",
        "hint": 'Your X-Plane Custom Scenery. Used for the automatic linking of built tiles (see auto_link_custom_scenery) and for "1-click" creation (or deletion) of symbolic links from Ortho4XP tiles to there.',
    },
    "custom_overlay_src": {
        "module": "OVL",
        "type": str,
        "default": "",
        "hint": "The directory containing the sceneries with the overlays you would like to extract. You need to select the level of directory just _ABOVE_ Earth nav data.",
    },
    "custom_overlay_src_alternate": {
        "module": "OVL",
        "type": str,
        "default": "",
        "hint": "If sceneries with overlays are not found in custom_overlay_src, set an alternate directory to search.",
    },
}

cfg_tile_vars = {
    # Vector
    "apt_smoothing_pix": {
        "type": int,
        "default": 8,
        "hint": "How much gaussian blur is applied to the elevation raster for the look up of altitude over airports. Unit is the elevation raster pixel size.",
    },
    "road_level": {
        "type": int,
        "default": 1,
        "values": (0, 1, 2, 3, 4, 5),
        "hint": 'Allows to level the mesh along roads and railways. Zero means nothing such is included; "1" looks for banking ways among motorways, primary and secondary roads and railway tracks; "2" adds tertiary roads; "3" brings residential and unclassified roads; "4" takes service roads, and 5 finishes with tracks. Purge the small_roads.osm cached data if you change your mind in between the levels 2-5.',
    },
    "road_banking_limit": {
        "type": float,
        "default": 0.5,
        "hint": "How much sloped does a roads need to be to be in order to be included in the mesh levelling process. The value is in meters, measuring the height difference between a point in the center of a road node and its closest point on the side of the road.",
    },
    "lane_width": {
        "type": float,
        "default": 4.0,
        "hint": "Width (in meters) to be used for buffering that part of the road network that requires leveling.",
    },
    "max_levelled_segs": {
        "type": int,
        "default": 200000,
        "hint": "This limits the total number of roads segments included for mesh levelling, in order to keep triangle count under control in case of abundant OSM data.",
    },
    "water_simplification": {
        "type": float,
        "default": 0.0,
        "hint": "In case the OSM data for water areas would become too large, this parameter (in meter) can be used for node simplification.",
    },
    "min_area": {
        "type": float,
        "default": 0.001,
        "hint": "Minimum area (in km^2) a water patch needs to be in order to be included in the mesh as such. Contiguous water patches are merged before area computation.",
    },
    "max_area": {
        "type": float,
        "default": 200.0,
        "hint": "Any water patch larger than this quantity (in km^2) will be masked like the sea.",
    },
    "clean_bad_geometries": {
        "type": bool,
        "default": True,
        "hint": "When set, all OSM geometries are checked for self-intersection and merged between themselves in case of overlapping, allowing (hopefully!) to go around most OSM errors. This is computationally expensive, especially in places where OSM road/water data is detailed, and this is the reason for this switch, but if you are not in a hurry it is probably wise leaving it always activated.",
    },
    "mesh_zl": {
        "type": int,
        "default": 19,
        "values": (16, 17, 18, 19, 20),
        "hint": "The mesh will be preprocessed to accept later any combination of imageries up to and including a zoomlevel equal to mesh_zl. Lower value could save a few tens of thousands triangles, but put a limitation on the maximum allowed imagery zoomlevel.",
    },
    # Mesh
    "curvature_tol": {
        "type": float,
        "default": 2.0,
        "hint": "This parameter is intrinsically linked the mesh final density. Mesh refinement is mostly based on curvature computations on the elevation data (the exact decision rule can be found in _ triunsuitable() _ in Utils/Triangle4XP.c). A higher curvature tolerance yields fewer triangles.",
    },
    "apt_curv_tol": {
        "type": float,
        "default": 0.5,
        "hint": "If smaller, it supersedes curvature_tol over airports neighbourhoods.",
    },
    "apt_curv_ext": {
        "type": float,
        "default": 0.5,
        "hint": "Extent (in km) around the airports where apt_curv_tol applies.",
    },
    "coast_curv_tol": {
        "type": float,
        "default": 1.0,
        "hint": "If smaller, it supersedes curvature_tol along the coastline.",
    },
    "coast_curv_ext": {
        "type": float,
        "default": 0.5,
        "hint": "Extent (in km) around the coastline where coast_curv_tol applies.",
    },
    "limit_tris": {
        "type": float,
        "default": 3.0,
        "hint": "If non zero, approx upper bound _in millions_ on the number of final triangles in the mesh. Note: When 0 we impose a hard limit of 5M, to keep X-Plane comfortable. For high resolution DEMS you _should_ use it.",
    },
    "min_angle": {
        "type": float,
        "default": 10.0,
        "hint": "The mesh algorithm will try to not have mesh triangles with (smallest for water / second smallest for regular land) angle less than the value (in deg) of min_angle.",
    },
    "sea_smoothing_mode": {
        "type": str,
        "default": "zero",
        "values": ["zero", "mean", "none"],
        "hint": "Zero means that all nodes of sea triangles are set to zero elevation. With mean, some kind of smoothing occurs (triangles are levelled one at a time to their mean elevation), None (a value mostly appropriate for DEM resolution of 10m and less), positive altitudes of sea nodes are kept intact, only negative ones are brought back to zero, this avoids to create unrealistic vertical cliffs if the coastline vector data was lower res.",
    },
    "water_smoothing": {
        "type": int,
        "default": 10,
        "hint": "Number of smoothing passes over all inland water triangles (sequentially set to their mean elevation).",
    },
    "iterate": {
        "type": int,
        "default": 0,
        "hint": "Allows to refine a mesh using higher resolution elevation data of local scope only (requires Gdal), typically LIDAR data. Having an iterate number is handy to go backward one step when some choice of parameters needs to be revised. REQUIRES cleaning_level=0.",
    },
    # Masks
    "mask_zl": {
        "type": int,
        "default": 14,
        "values": (14, 15, 16),
        "hint": "The zoomlevel at which the (sea) water masks are built. Masks are used for alpha channel, and this channel usually requires less resolution than the RGB ones, the reason for this (VRAM saving) parameter. If the coastline and elevation data are very detailed, it might be interesting to lift this parameter up so that the masks can reproduce this complexity.",
    },
    "masks_width": {
        "type": list,
        "default": 100,
        "hint": "Maximum extent of the masks perpendicularly to the coastline (rough definition). NOTE: The value is now in meters, it used to be in ZL14 pixel size in earlier verions, the scale is roughly one to ten between both.",
    },
    "masking_mode": {
        "type": str,
        "default": "sand",
        "values": ["sand", "rocks", "3steps"],
        "hint": 'A selection of three tentative masking algorithms (still looking for the Holy Grail...). The first two (sand and rocks) requires masks_width to be a single value; the third one (3steps) requires a list of the form [a,b,c] for masks width: "a" is the length in meters of a first transition from plain imagery at the shoreline towards ratio_water transparency, "b" is the second extent zone where transparency level is kept constant equal to ratio_water, and "c" is the last extent where the masks eventually fade to nothing. The transition with rocks is more abrupt than with sand.',
    },
    "use_masks_for_inland": {
        "type": bool,
        "default": False,
        "hint": "Will use masks for the inland water (lakes, rivers, etc) too, instead of the default constant transparency level determined by ratio_water. This is VRAM expensive and presumably not really worth the price.",
    },
    "imprint_masks_to_dds": {
        "type": bool,
        "default": False,
        "hint": "Will apply masking directly to dds textures (at the Build Imagery/DSF step) rather than using external png files. This doubles the file size of masked textures (dxt5 vs dxt1) but reduce the overall VRAM footprint (a matter of choice!)",
    },
    "distance_masks_too": {
        "type": bool,
        "default": False,
        "hint": "This will additionally build distance to coastline masks that are used in Step 3 in order to improve the bathymetric profile (otherwise too low res) and avoid steep walls close to piers or rocks. Masks_zl should not be too low to grab these details.",
    },
    "masks_use_DEM_too": {
        "type": bool,
        "default": False,
        "hint": "If you have acces to high resolutions DEMs (really shines with 5m or lower), you can use the elevation in addition to the vector data in order to draw masks with higher precision. If the DEM is not high res, this option will yield unpleasant pixellisation.",
    },
    "masks_custom_extent": {
        "type": str,
        "default": "",
        "hint": 'Yet another tentative to draw masks with maximizing the use of the good imagery part. Requires to draw (JOSM) the "good imagery" threshold first, but it could be one order of magnitude faster to do compared to hand tweaking the masks and the imageries one by one.',
    },
    # DSF/Imagery
    "default_website": {"type": str, "default": "", "hint": ""},
    "default_zl": {"type": int, "default": 16, "hint": ""},
    "zone_list": {"type": list, "default": [], "hint": ""},
    "cover_airports_with_highres": {
        "type": str,
        "default": "False",
        "values": ("False", "True", "ICAO", "Existing"),
        "hint": 'When set, textures above airports will be upgraded to a higher zoomlevel, the imagery being the same as the one they would otherwise receive. Can be limited to airports with an ICAO code for tiles with so many airports. Exceptional: use "Existing" to (try to) derive custom zl zones from the textures directory of an existing tile.',
        "short_name": "high_zl_airports",
    },
    "cover_extent": {
        "type": float,
        "default": 1.0,
        "hint": "The extent (in km) past the airport boundary taken into account for higher ZL. Note that for VRAM efficiency higher ZL textures are fully used on their whole extent as soon as part of them are needed.",
    },
    "cover_zl": {
        "type": int,
        "default": 18,
        "hint": "The zoomlevel with which to cover the airports zone when high_zl_airports is set. Note that if the cover_zl is lower than the zoomlevel which would otherwise be applied on a specific zone, the latter is used.",
    },
    "airport_zone_icaos": {
        "type": str,
        "default": "",
        "hint": "Comma-separated ICAO codes of airports (e.g. RCKH,RCAY) that get ZL19, ZL18 and ZL17 zones added to zone_list automatically around their runways, at the distances set by airport_zone_zl19_m / _zl18_m / _zl17_m. An airport only counts for the tile its runways are in; listed in another tile it is skipped. Where its zones reach into neighbouring tiles of the same batch build, those tiles get them too. Drawn zones of a higher zoomlevel still win over them. Runways come from apt.dat, so custom_scenery_dir must be set. Tile setting only.",
        "short_name": "apt_zone_icaos",
    },
    "airport_zone_zl19_m": {
        "type": float,
        "default": 1600.0,
        "hint": "Distance (m) from the polygon spanning all runways of each airport in airport_zone_icaos to the edge of its ZL19 zone. 0 leaves the ZL19 zone out.",
        "short_name": "apt_zone_zl19_m",
    },
    "airport_zone_zl18_m": {
        "type": float,
        "default": 4350.0,
        "hint": "Distance (m) from the polygon spanning all runways of each airport in airport_zone_icaos to the edge of its ZL18 zone. 0 leaves the ZL18 zone out.",
        "short_name": "apt_zone_zl18_m",
    },
    "airport_zone_zl17_m": {
        "type": float,
        "default": 14550.0,
        "hint": "Distance (m) from the polygon spanning all runways of each airport in airport_zone_icaos to the edge of its ZL17 zone. 0 leaves the ZL17 zone out.",
        "short_name": "apt_zone_zl17_m",
    },
    "color_reference_year": {
        "type": int,
        "default": 2025,
        "hint": "For the Colour correction and Cloud removal batch steps: which yearly Sentinel-2 cloudless mosaic (EOX) the imagery is matched against. Years differ in season and haze per place, so if one looks wrong for a tile try another. Saving the config checks the year with the EOX server. Textures corrected against another year are redone on the next build.",
    },
    "sea_texture_blur": {
        "type": float,
        "default": 0.0,
        "hint": 'For layers of type "mask" in combined providers imageries, determines the extent (in meters) of the blur radius applied. This allows to smoothen some sea imageries where the wave or reflection pattern was too much present.',
    },
    "water_tech": {
        "type": str,
        "default": "XP11 + bathy",
        "values": ("XP12", "XP11 + bathy"),
        "hint": "Water tech type. XP12 uses a new (partly in construction) rendering tech, XP11 + bathy uses a more traditionnal blend. Both allows for 3D water.",
    },
    # "add_low_res_sea_ovl": {
    #    "type": bool,
    #    "default": False,
    #    "hint": "Will add an extra texture layer over the sea (with constant alpha channel given by ratio_water as for inland water), based on a low resolution imagery with global coverage. Masks with their full resolution imagery are still being used when present, the final render is a composite of both. The default imagery with code SEA can be changed as any other imagery defined in the Providers directory, it needs to have a max_zl defined and is used at its max_zl.",
    # },
    # "experimental_water": {
    #    "type": int,
    #    "default": 0,
    #    "values": (0, 1, 2, 3),
    #    "hint": 'If non zero, replaces X-Plane water by a custom normal map over low res ortho-imagery (requires XP11 but turns water rendering more XP10 alike). The value 0 corresponds to legacy X-Plane water, 1 replaces it for inland water only, 2 over sea water only, and 3 over both. Values 2 and 3 should always be used in combination with "imprint_masks_to_dds".\n\nThis experimental feature has two strong downsides: 1) the waves are static rather dynamical (would require a plugin to update the normal_map as X-Plane does) and 2) the wave height is no longer weather dependent. On the other hand, waves might have less repetitive patterns and some blinking in water reflections might be improved too; users are welcome to improve the provided water_normal_map.dds (Gimp can be used to edit the mipmaps individually).',
    # },
    "ratio_water": {
        "type": float,
        "default": 0.25,
        "hint": 'Inland water rendering is made of two layers : one bottom layer of "X-Plane water" and one overlay layer of orthophoto with constant level of transparency applied. The parameter ratio_water (values between 0 and 1) determines how much transparency is applied to the orthophoto. At zero, the orthophoto is fully opaque and X-Plane water cannot be seen ; at 1 the orthophoto is fully transparent and only the X-Plane water is seen.',
    },
    "ratio_bathy": {
        "type": float,
        "default": 1.0,
        "hint": "Bathymetry multiplier for near shore vertices. In the range [0,1].",
    },
    "normal_map_strength": {
        "type": float,
        "default": 1.0,
        "hint": 'Orthophotos by essence already contain the part of the shading burned in (here by shading we mean the amount of reflected light in the camera direction as a function of the terrain slope, not the shadows). This option allows to tweak the normal coordinates of the mesh in the DSF to avoid "overshading", but it has side effects on the way X-Plane computes scenery shadows. Used to be 0.3 by default in earlier versions, the default is now 1 which means exact normals.',
    },
    "terrain_casts_shadows": {
        "type": bool,
        "default": True,
        "hint": "If unset, the terrain itself will not cast (but still receive!) shadows. This option is only meaningful if scenery shadows are opted for in the X-Plane graphics settings.",
        "short_name": "terrain_casts_shadow",
    },
    "overlay_lod": {
        "type": float,
        "default": 25000,
        "hint": "Distance until which overlay imageries (that is orthophotos over water) are drawn. Lower distances have a positive impact on frame rate and VRAM usage, and IFR flyers will probably need a higher value than VFR ones.",
    },
    "use_decal_on_terrain": {
        "type": bool,
        "default": False,
        "hint": "Terrain files for all but water triangles will contain a detail decal directive (see terrain_decal). The effect is noticeable at very low altitude and helps to overcome the orthophoto blur at such levels. Can be slightly distracting at higher altitude.",
    },
    "terrain_decal": {
        "type": str,
        "default": "o4xp",
        "values": ["o4xp", "xplane"],
        "hint": "Decal used by use_decal_on_terrain. xplane is X-Plane's maquify_2_green_key.dcl, one 25 m texture whose repeat shows as a grid over fields. o4xp is the same shrub/dirt detail re-synthesised into two non-repeating layers (100 m and 161.8 m) so no grid forms; it needs the O4XP_Decal_Library scenery pack installed (scripts/asset_pipeline/decals).",
    },
    "terrain_materials": {
        "type": bool,
        "default": True,
        "hint": "X-Plane 12 material map for every land texture: a <texture>_mat.dds next to it (gloss, base reflectance and a faint micro-relief normal, estimated from the orthophoto colours: vegetation stays matte, asphalt, concrete and roofs get a little sheen) referenced from the texture's .ter file. Textures kept from an earlier build get theirs on the next Step 3. Costs about 2 s per ZL17+ texture (5.6 MB map) and under 1 s per ZL16 texture (1.4 MB map).",
    },
    "terrain_materials_relief": {
        "type": float,
        "default": 0.15,
        "hint": "Strength of the micro-relief in the terrain material maps (terrain_materials), taken from the fine detail of the orthophoto: 0.15 tilts the surface by about 5 degrees on average, 0 keeps it flat. Changing it remakes the maps on the next Step 3.",
    },
    # Vegetation overlay (src/scripts/generate_veg_overlay.py)
    "veg_density": {
        "type": float,
        "default": -1.0,
        "short_name": "veg_den",
        "hint": "DSF density override 0.0–1.0 (scaled to 0–255). -1 = auto-select based on how much of each polygon is covered by vegetation.",
    },
    "veg_close_m": {
        "type": float,
        "default": 10.0,
        "short_name": "veg_close",
        "hint": "Morphological close radius in metres. Fills canopy gaps within a tree patch. Larger values merge nearby trees into one polygon; smaller values preserve fine detail.",
    },
    "veg_open_m": {
        "type": float,
        "default": 3.0,
        "short_name": "veg_open",
        "hint": "Morphological open radius in metres. Removes tiny isolated detections before contouring. Raise to suppress small spurious tree patches.",
    },
    "veg_min_area_m2": {
        "type": float,
        "default": 50.0,
        "short_name": "veg_min_a",
        "hint": "Minimum vegetation polygon area in m² to include in the DSF. Raise to reduce polygon count; lower to capture small isolated trees.",
    },
    "veg_simplify_m": {
        "type": float,
        "default": 3.0,
        "short_name": "veg_simp",
        "hint": "Douglas-Peucker polygon simplification tolerance in metres. Higher values produce fewer, coarser vertices; lower values preserve exact canopy outlines.",
    },
    "veg_excl_buffer_m": {
        "type": float,
        "default": 5.0,
        "short_name": "veg_excl",
        "hint": "Dilation buffer in metres applied around detected roads and buildings before subtracting them from the tree mask. Prevents trees bleeding onto building edges or road surfaces.",
    },
    "veg_use_simheaven": {
        "type": bool,
        "default": True,
        "short_name": "veg_simh",
        "hint": "Use simHeaven X-World network roads for road-width tree exclusion. This setting is only for road masks; simHeaven building overlap is controlled separately by the simh_bd / simh_bd_b settings.",
    },
    "veg_avoid_simheaven_buildings": {
        "type": bool,
        "default": True,
        "short_name": "veg_simh_bd",
        "hint": "Avoid overlapping simHeaven building footprints and objects when generating vegetation. This is separate from the building overlay's own placement exclusion.",
    },
    "veg_simheaven_building_buffer_m": {
        "type": float,
        "default": 10.0,
        "short_name": "veg_simh_bd_b",
        "hint": "Extra exclusion buffer in metres around simHeaven building footprints and objects.",
    },
    "veg_avoid_simheaven_forests": {
        "type": bool,
        "default": True,
        "short_name": "veg_simh_for",
        "hint": "Avoid overlapping polygons from enabled simHeaven X-World 7-forests packages.",
    },
    "veg_simheaven_forest_buffer_m": {
        "type": float,
        "default": 0.0,
        "short_name": "veg_simh_for_b",
        "hint": "Extra exclusion buffer in metres around simHeaven X-World forest polygons.",
    },
    "veg_exclude_lower_forests": {
        "type": bool,
        "default": False,
        "short_name": "veg_excl_for",
        "hint": "Replace lower-priority forests throughout this tile. The "
                "generated vegetation DSF carries a sim/exclude_for property "
                "covering the WHOLE one-degree tile, so every forest placed by "
                "a lower-priority package (simHeaven, SwissUHD, ...) disappears "
                "across the entire tile — including areas where the model "
                "places no trees at all. Higher-priority packs such as airports "
                "are unaffected. Off by default; turn it on only for tiles you "
                "intend to re-vegetate completely.",
    },
    "veg_remove_embedded_trees": {
        "type": bool,
        "default": False,
        "short_name": "veg_rm_trees",
        "hint": "Also remove supported embedded trees from lower-priority "
                "scenery in this tile. sim/exclude_for only suppresses forest "
                "polygons; trees baked into autogen blocks, road vegetation or "
                "standalone tree objects survive it. This edits the affected "
                "source DSFs in place, keeping a verified backup of every "
                "original, and needs veg_exclude_lower_forests enabled. "
                "Unticking it later does NOT undo installed edits — use "
                "the Restore Veg Tile action for that.",
    },
    "veg_use_simheaven_asset_proximity": {
        "type": bool,
        "default": True,
        "short_name": "veg_simh_typ",
        "hint": "Choose broadleaf, conifer, or mixed assets from the nearest "
                "simHeaven forest polygon, with the tile's most common tree "
                "family as fallback.",
    },
    "veg_use_leaf_type_maps": {
        "type": bool,
        "default": True,
        "short_name": "veg_leaf_maps",
        "hint": "Choose broadleaf, conifer, or mixed assets from leaf-type maps "
                "(Copernicus HRL 10 m in Europe, GLC_FCS30D 30 m and CGLS "
                "100 m worldwide, explicit OSM leaf_type) and split forests "
                "along their boundaries. Map data downloads once per tile "
                "(once per 5x5 degrees for GLC_FCS30D, 10-700 MB) into "
                "Overlay_cache/_leaf_type. Falls back to the simHeaven closest "
                "asset where no map covers the tile.",
    },
    "veg_use_xp_forests": {
        "type": bool,
        "default": True,
        "short_name": "veg_xp_for",
        "hint": "Plant X-Plane 12's own climate forests (lib/g8 broad/coni/"
                "mixed per temperature and moisture band, orchard sets in "
                "orchards) instead of the three simHeaven forest aliases. The "
                "band follows the default global scenery's nearest forest, "
                "with Koppen climate as fallback.",
    },
    "veg_canopy": {
        "type": bool,
        "default": True,
        "short_name": "veg_canopy",
        "hint": "Measure canopy height on ZL18 and ZL19 textures (Meta/WRI "
                "canopy-height model). Inside airport boundaries it decides "
                "where trees are, instead of the land-cover classes, which read "
                "airfield grass as trees; everywhere on those textures it sizes "
                "the trees (veg_tree_sizes, veg_forest_heights). Needs CUDA; the "
                "3.1 GB model downloads once into Overlay_cache/_canopy_height.",
    },
    "veg_canopy_min_conf": {
        "type": float,
        "default": 0.2,
        "short_name": "veg_can_conf",
        "hint": "ZL18/19: a pixel is a tree when the canopy model gives at "
                "least this probability to it being 2 m or taller. Lower "
                "plants more trees (0.15 fills gardens, under 0.1 overfills), "
                "higher plants fewer (0.3 and up starts dropping real woods). "
                "0 uses the model's mean height >= 2 m instead.",
    },
    "veg_canopy_seg_min_h_m": {
        "type": float,
        "default": 0.25,
        "short_name": "veg_can_seg",
        "hint": "ZL18/19: also keep the land-cover model's trees where the "
                "canopy is at least this tall, in metres. Brings back low "
                "scrub, young plantations and bush clusters that the canopy "
                "model rates under 2 m, while flat grass (~0 m) stays bare. "
                "0 turns it off.",
    },
    "veg_forest_holes": {
        "type": bool,
        "default": True,
        "short_name": "veg_holes",
        "hint": "Cut fields, yards and meadows enclosed by a forest out of it "
                "as holes, so X-Plane leaves them bare instead of spreading "
                "the forest's trees across the whole outline.",
    },
    "veg_point_tree_max_cells": {
        "type": float,
        "default": 2.0,
        "short_name": "veg_pt_cells",
        "hint": "Tree patches smaller than this many grid cells of their "
                "forest become single trees, one per crown. A cell is the "
                "forest's tree spacing squared (10-12 m for X-Plane 12 "
                "forests); X-Plane's grid often plants no tree at all in "
                "a polygon under 2 cells. Patches under 600 m2 always do.",
    },
    "veg_hole_certain_cells": {
        "type": float,
        "default": 1.0,
        "short_name": "veg_hole_sure",
        "hint": "Cut a clearing out of a forest from this many grid cells "
                "when its cause is known: a building, road, water, field, "
                "bare ground, flat canopy, or another forest type.",
    },
    "veg_hole_unsure_cells": {
        "type": float,
        "default": 0.0,
        "short_name": "veg_hole_unsure",
        "hint": "Cut a clearing out of a forest from this many grid cells "
                "when it is only ground the models were not confident is "
                "tree (0 = never: such gaps stay forest).",
    },
    "veg_forest_hole_min_m2": {
        "type": float,
        "default": 200.0,
        "short_name": "veg_hole_m2",
        "hint": "Replaced by veg_hole_certain_cells / veg_hole_unsure_cells; "
                "kept so older config files still load.",
    },
    "veg_tree_sizes": {
        "type": bool,
        "default": True,
        "short_name": "veg_tree_sizes",
        "hint": "ZL18/19: plant single trees as X-Plane's sized single-species "
                "trees (small to tall oak, pine, kapok... as X-Plane plants in "
                "that climate), sized to the canopy height measured at each "
                "crown, instead of the full-height climate forest. ZL16/17 "
                "imagery is too coarse to measure heights on.",
    },
    "veg_forest_heights": {
        "type": bool,
        "default": True,
        "short_name": "veg_forest_h",
        "hint": "ZL18/19: give forests whose measured canopy is well under "
                "X-Plane's climate forest a copy of that forest with shorter "
                "trees (canopy 6, 9, 12 or 15 m), and forests measured under 5 m "
                "(X-Plane's smallest tree) its shrubs of the nearest size. The "
                "copies are written into the O4XP_Forest_Heights library pack in "
                "Custom Scenery.",
    },
    "veg_approach_heights": {
        "type": bool,
        "default": True,
        "short_name": "veg_approach_h",
        "hint": "Keep trees under runway approaches below the ICAO approach "
                "surface: from the inner edge 30-60 m before the threshold the "
                "allowed height rises with the surface's slope (2-5 % by runway "
                "length and approach type), so trees get gradually shorter "
                "towards the runway; the tallest trees fit from about 2.3 km "
                "out. Under 9 m they become shrubs of the size that fits, and "
                "nearest the threshold 1 m shrubs. Uses forest copies in the "
                "O4XP_Forest_Heights pack.",
    },
    "veg_osm_rescue_rangeland": {
        "type": bool,
        "default": True,
        "short_name": "veg_osm_range",
        "hint": "Inside OpenStreetMap forests (simHeaven 7-forests), plant "
                "trees where SegFormer sees rangeland that looks like canopy in "
                "the ortho (dark green): it calls whole wooded hillsides "
                "rangeland. Grass, scrub, bare ground, buildings and roads "
                "stay open.",
    },
    "veg_upscale_gap_trees": {
        "type": bool,
        "default": True,
        "short_name": "veg_upscale_trees",
        "hint": "ZL16: run SegFormer a second time on each texture enlarged 2x "
                "and add the trees it finds there: tree clumps and orchards "
                "between fields and garden and street trees that the normal "
                "pass gives to the fields or houses. About 4x the vegetation "
                "inference time on ZL16 textures.",
    },
    "veg_tropics_evergreen": {
        "type": bool,
        "default": True,
        "short_name": "veg_tropics_ev",
        "hint": "Between the tropics, plant X-Plane's seasonal forests and shrubs "
                "as evergreen copies of their summer look. X-Plane runs its "
                "seasons by the hemisphere's calendar, so on the equator its "
                "temperate-band forests (planted on tropical highlands) turn "
                "autumn brown in October. Uses the O4XP_Forest_Heights pack.",
    },
    "veg_avoid_gfv2": {
        "type": bool,
        "default": False,
        "short_name": "veg_gfv2",
        "hint": "Legacy compatibility setting. Global Forests is no longer read by the vegetation pipeline.",
    },
    "veg_gfv2_buffer_m": {
        "type": float,
        "default": 0.0,
        "short_name": "veg_gfv2_b",
        "hint": "Legacy compatibility setting. Global Forests is no longer read by the vegetation pipeline.",
    },
    "veg_use_gfv2_asset_proximity": {
        "type": bool,
        "default": False,
        "short_name": "veg_gfv2_typ",
        "hint": "Legacy compatibility setting. New vegetation always uses simHeaven X-World forest assets.",
    },
    "veg_res_m": {
        "type": float,
        "default": 0.0,
        "short_name": "veg_res",
        "hint": "Downsample inference maps to this ground resolution in metres before polygon extraction. 0 = native DDS detail (about 2 m at ZL16). Use 5–10 m to reduce polygon count on performance-critical tiles.",
    },
    "grass_enabled": {
        "type": bool,
        "default": True,
        "short_name": "grass",
        "hint": "Plant 3-d grass around every airport in the tile, coloured "
                "from the ortho underneath. Runs as part of the vegetation "
                "overlay and goes into the same DSF. Grass grows inside each "
                "airport's apt.dat boundary (or, without one, a circle out to "
                "its furthest runway or taxiway point) plus the buffer below, "
                "never on runways, taxiways, aprons, roads or rail, and in "
                "desert climates only where the ortho is green.",
    },
    "grass_airport_buffer_m": {
        "type": float,
        "default": 500.0,
        "short_name": "grass_buf",
        "hint": "How far the grass reaches beyond each airport's boundary, in "
                "metres.",
    },
    "grass_full_tile": {
        "type": bool,
        "default": False,
        "short_name": "grass_tile",
        "hint": "Grow grass over the whole tile instead of only around its "
                "airports (same rules: not on runways, roads, rail or built "
                "ground). Only the grass within 230 m of the camera is drawn, "
                "but the grass DSF, the build time and the disk used while "
                "building grow with the area covered.",
    },
    "veg_disable_cache": {
        "type": bool,
        "default": False,
        "short_name": "veg_no_cache",
        "hint": "Disable persistent per-DDS vegetation cache files for this tile. When enabled, vegetation cache files are not reused and any transient per-texture cache written during processing is deleted immediately after that texture is finished.",
    },
    "veg_verbose_log": {
        "type": bool,
        "default": False,
        "short_name": "veg_verbose_log",
        "hint": "Show vegetation setup, scenery scans, cache diagnostics and timing breakdowns. When disabled, keep texture progress, final results, warnings and errors.",
    },
    "segformer_patch_size": {
        "type": int,
        "default": 512,
        "short_name": "seg_patch",
        "hint": "SegFormer inference patch size. 512 matches the model's training resolution and covers roughly 1 km of ZL16 imagery edge to edge. Larger values need more VRAM; smaller may reduce accuracy at patch boundaries.",
    },
    "segformer_overlap": {
        "type": int,
        "default": 64,
        "short_name": "segformer_overlap",
        "hint": "Overlap between adjacent inference patches. Higher values reduce seam artifacts at patch boundaries at the cost of more inference passes. 64 is a good balance for 512-size patches and is roughly 120 m on ZL16 imagery.",
    },
    "segformer_batch_size": {
        "type": int,
        "default": 0,
        "short_name": "seg_batch",
        "hint": "SegFormer inference batch size. 0 = recommended default (8 on CUDA, 1 on CPU). Raise only after timing a representative tile.",
    },
    # Building overlay (src/scripts/generate_bld_overlay.py)
    "bld_spacing_m": {
        "type": float,
        "default": 0.0,
        "short_name": "bld_spacing_m",
        "hint": "Minimum target gap in metres between building footprint edges. Candidate centre spacing is derived from this gap plus the smallest asset footprint span in each size class.",
    },
    "bld_close_m": {
        "type": float,
        "default": 30.0,
        "short_name": "bld_gap_m",
        "hint": "Gap-closing radius in metres for building zones. Larger values bridge wider gaps between nearby building detections.",
    },
    "bld_open_m": {
        "type": float,
        "default": 10.0,
        "short_name": "bld_cleanup_m",
        "hint": "Cleanup radius in metres for building zones. Larger values remove more tiny isolated detections.",
    },
    "bld_min_footprint_m2": {
        "type": float,
        "default": 12.0,
        "short_name": "bld_min_footprint_m2",
        "hint": "Minimum building footprint area in square metres. Individual detections smaller than this are discarded up front, before any placement. Building zones are otherwise accepted regardless of size.",
    },
    "bld_grid_n": {
        "type": int,
        "default": 16,
        "short_name": "bld_grid",
        "hint": "Heading-grid resolution per texture tile. Higher values follow street direction changes more closely but can make orientation noisier.",
    },
    "bld_disable_cache": {
        "type": bool,
        "default": False,
        "short_name": "bld_no_cache",
        "hint": "Disable persistent per-DDS building cache files for this tile. When enabled, building cache files are not reused and any transient per-texture cache written during processing is deleted immediately after that texture is finished.",
    },
    "bld_verbose_log": {
        "type": bool,
        "default": False,
        "short_name": "bld_verbose_log",
        "hint": "Show building setup, asset and scenery scans, cache diagnostics, placement statistics and timing breakdowns. When disabled, keep one summary per texture, final results, warnings and errors.",
    },
    "bld_avoid_custom_scenery": {
        "type": bool,
        "default": True,
        "short_name": "bld_avoid_custom",
        "hint": "Avoid overlapping enabled Custom Scenery objects and facade polygons from scenery_packs.ini when generating building overlays.",
    },
    "bld_asset_mode": {
        "type": str,
        "default": "both",
        "values": ["objects", "facades", "both"],
        "short_name": "bld_assets",
        "hint": "Which building placement assets to use: object models, scalable facades, or object-first placement with facade fallback.",
    },
    "bld_roof_color_matching": {
        "type": bool,
        "default": False,
        "short_name": "bld_roof_color",
        "hint": "Prefer object assets whose analyzed roof texture color matches each detected rooftop. Disabled by default. Existing fit, containment, occupancy, height, and fallback selection rules remain authoritative.",
    },
    "bld_yolo_enabled": {
        "type": bool,
        "default": True,
        "short_name": "bld_yolo",
        "hint": "Use the YOLO OBB building-placement model as the primary source of building object placements before filling remaining SegFormer building-zone gaps.",
    },
    "bld_yolo_checkpoint": {
        "type": str,
        "default": r"H:\model_training\runs\yolo_obb_v1\weights\visual_candidate_step_12000.pt",
        "short_name": "bld_yolo_ckpt",
        "file_picker": True,
        "file_picker_title": "Choose YOLO model checkpoint",
        "filetypes": [
            ("YOLO model files", ("*.pt", "*.pth", "*.onnx", "*.engine")),
            ("All files", "*.*"),
        ],
        "hint": "YOLO OBB checkpoint used for primary building placements.",
    },
    "bld_yolo_conf": {
        "type": float,
        "default": 0.18,
        "short_name": "bld_yolo_conf",
        "hint": "YOLO OBB confidence threshold for direct building placements.",
    },
    "bld_yolo_iou": {
        "type": float,
        "default": 0.5,
        "short_name": "bld_yolo_iou",
        "hint": "YOLO OBB NMS IoU threshold for direct building placements.",
    },
    "bld_yolo_stride": {
        "type": int,
        "default": 512,
        "short_name": "bld_yolo_stride",
        "hint": "YOLO OBB inference crop stride in pixels.",
    },
    "bld_yolo_max_det": {
        "type": int,
        "default": 100000,
        "short_name": "bld_yolo_max",
        "hint": "YOLO OBB maximum detections per inference crop.",
    },
    "bld_height_checkpoint": {
        "type": str,
        "default": r"H:\model_training\models\heightnet.pt",
        "short_name": "bld_height_ckpt",
        "file_picker": True,
        "file_picker_title": "Choose HeightNet model checkpoint",
        "filetypes": [
            ("HeightNet model files", ("*.pt", "*.pth")),
            ("All files", "*.*"),
        ],
        "hint": "HeightNet checkpoint used to predict per-building heights from "
                "YOLO detections (second stage). Leave at the default unless you "
                "trained a replacement. If the file is missing, buildings fall "
                "back to class-default heights.",
    },
    "bld_yolo_min_coverage": {
        "type": float,
        "default": 0.80,
        "short_name": "bld_yolo_min_cov",
        "hint": "Minimum fraction of a YOLO detection polygon that a 3D object's "
                "footprint must cover to be placed. Lower it to admit smaller objects "
                "that fully fit when no near-detection-size object exists (small "
                "residential classes already use lower per-class floors). Objects never "
                "overhang the polygon regardless of this value.",
    },
    # Other
    "custom_dem": {
        "type": str,
        "default": "",
        "hint": "Path to an elevation data file to be used instead of the default Viewfinderpanoramas.org ones (J. de Ferranti). The raster must be in geopgraphical coordinates (EPSG:4326) but the extent need not match the tile boundary (requires Gdal). Regions of the tile that are not covered by the raster are mapped to zero altitude (can be useful for high resolution data over islands in particular).",
    },
    "fill_nodata": {
        "type": bool,
        "default": True,
        "hint": "When set, the no_data values in the raster will be filled by a nearest neighbour algorithm. If unset, they are turned into zero (can be useful for rasters with no_data over the whole oceanic part or partial LIDAR data).",
    },
}

# Settings that only exist per tile: the global config has no global_ copy.
tile_only_vars = [
    "default_website", "default_zl", "zone_list", "airport_zone_icaos",
]

# Create dictionary from cfg_tile_vars with prefix and remove keys not in global config
cfg_global_tile_vars = {
    f"{global_prefix}{key}": value
    for key, value in cfg_tile_vars.items()
    if key not in tile_only_vars
}

cfg_vars = {**cfg_app_vars, **cfg_tile_vars, **cfg_global_tile_vars}

list_app_vars = [
    "verbosity",
    "cleaning_level",
    "ui_theme",
    "overpass_server_choice",
    "skip_downloads",
    "skip_converts",
    "max_download_slots",
    "max_convert_slots",
    "check_tms_response",
    "http_timeout",
    "max_connect_retries",
    "max_baddata_retries",
    "ovl_exclude_pol",
    "ovl_exclude_net",
    "auto_link_custom_scenery",
    "custom_scenery_dir",
    "custom_overlay_src",
    "custom_overlay_src_alternate",
    "osm_pbf_dir",
]

gui_app_vars_short = list_app_vars[:-4]

gui_app_vars_long = list_app_vars[-4:]

list_vector_vars = [
    "apt_smoothing_pix",
    "road_level",
    "road_banking_limit",
    "lane_width",
    "max_levelled_segs",
    "water_simplification",
    "min_area",
    "max_area",
    "clean_bad_geometries",
    "mesh_zl",
]

list_mesh_vars = [
    "curvature_tol",
    "apt_curv_tol",
    "apt_curv_ext",
    "coast_curv_tol",
    "coast_curv_ext",
    "limit_tris",
    "min_angle",
    "sea_smoothing_mode",
    "water_smoothing",
    "iterate",
]

list_mask_vars = [
    "mask_zl",
    "masks_width",
    "masking_mode",
    "use_masks_for_inland",
    "imprint_masks_to_dds",
    "distance_masks_too",
    "masks_use_DEM_too",
    "masks_custom_extent",
]

list_dsf_vars = [
    "cover_airports_with_highres",
    "cover_extent",
    "cover_zl",
    "airport_zone_icaos",
    "airport_zone_zl19_m",
    "airport_zone_zl18_m",
    "airport_zone_zl17_m",
    "color_reference_year",
    "water_tech",
    "ratio_bathy",
    "ratio_water",
    "overlay_lod",
    "sea_texture_blur",
    # "add_low_res_sea_ovl",
    # "experimental_water",
    "normal_map_strength",
    "terrain_casts_shadows",
    "use_decal_on_terrain",
    "terrain_decal",
    "terrain_materials",
    "terrain_materials_relief",
]

list_other_vars = ["custom_dem", "fill_nodata"]

list_veg_vars = [
    "veg_density",
    "veg_close_m",
    "veg_open_m",
    "veg_min_area_m2",
    "veg_simplify_m",
    "veg_excl_buffer_m",
    "veg_use_simheaven",
    "veg_avoid_simheaven_buildings",
    "veg_simheaven_building_buffer_m",
    "veg_avoid_simheaven_forests",
    "veg_simheaven_forest_buffer_m",
    "veg_exclude_lower_forests",
    "veg_remove_embedded_trees",
    "veg_use_simheaven_asset_proximity",
    "veg_use_leaf_type_maps",
    "veg_use_xp_forests",
    "veg_canopy",
    "veg_canopy_min_conf",
    "veg_canopy_seg_min_h_m",
    "veg_forest_holes",
    "veg_point_tree_max_cells",
    "veg_hole_certain_cells",
    "veg_hole_unsure_cells",
    "veg_tree_sizes",
    "veg_forest_heights",
    "veg_approach_heights",
    "veg_tropics_evergreen",
    "veg_upscale_gap_trees",
    "veg_osm_rescue_rangeland",
    "veg_res_m",
    "grass_enabled",
    "grass_airport_buffer_m",
    "grass_full_tile",
    "veg_disable_cache",
    "veg_verbose_log",
    "segformer_patch_size",
    "segformer_overlap",
    "segformer_batch_size",
]

# Keep removed Global Forests settings loadable so old tile/global config files
# remain valid and code that copies a Tile object does not lose attributes. They
# are intentionally excluded from the vegetation settings UI above.
list_veg_legacy_vars = [
    "veg_forest_hole_min_m2",
    "veg_avoid_gfv2",
    "veg_gfv2_buffer_m",
    "veg_use_gfv2_asset_proximity",
]

list_bld_vars = [
    "bld_spacing_m",
    "bld_close_m",
    "bld_open_m",
    "bld_min_footprint_m2",
    "bld_grid_n",
    "bld_disable_cache",
    "bld_verbose_log",
    "bld_avoid_custom_scenery",
    "bld_asset_mode",
    "bld_roof_color_matching",
    "bld_yolo_enabled",
    "bld_yolo_checkpoint",
    "bld_yolo_conf",
    "bld_yolo_iou",
    "bld_yolo_stride",
    "bld_yolo_max_det",
    "bld_yolo_min_coverage",
    "bld_height_checkpoint",
]

list_overlay_vars = list_veg_vars + list_bld_vars

_segformer_vars = ["segformer_patch_size", "segformer_overlap", "segformer_batch_size"]

# What O4_Overlay_Pipeline.apply_tile_settings() copies from a Tile before each
# overlay step; the GUI single-tile buttons and batch builds share these lists.
list_veg_pipeline_vars = list_veg_vars + list_veg_legacy_vars
list_bld_pipeline_vars = list_bld_vars + _segformer_vars

list_tile_vars = (
    list_vector_vars
    + list_mesh_vars
    + list_mask_vars
    + list_dsf_vars
    + list_overlay_vars
    + list_veg_legacy_vars
    + list_other_vars
    + ["default_website", "default_zl", "zone_list"]
)

list_global_tile_vars = [
    global_prefix + item
    for item in (
        list_vector_vars
        + list_mesh_vars
        + list_mask_vars
        + list_dsf_vars
        + list_overlay_vars
        + list_veg_legacy_vars
        + list_other_vars
    )
    if item not in tile_only_vars
]

list_global_vector_vars = [global_prefix + item for item in list_vector_vars]

list_global_mesh_vars = [global_prefix + item for item in list_mesh_vars]

list_global_dsf_vars = [
    global_prefix + item for item in list_dsf_vars if item not in tile_only_vars
]

list_global_mask_vars = [global_prefix + item for item in list_mask_vars]

list_global_veg_vars = [global_prefix + item for item in list_veg_vars]

list_global_bld_vars = [global_prefix + item for item in list_bld_vars]

list_global_overlay_vars = [global_prefix + item for item in list_overlay_vars]

list_cfg_vars = list_tile_vars + list_global_tile_vars + list_app_vars
