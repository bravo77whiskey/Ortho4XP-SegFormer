import os
import time
import io
import bz2
import random
import requests
import numpy
from shapely import geometry, ops
import O4_UI_Utils as UI
import O4_File_Names as FNAMES
import O4_PBF_Utils as PBF
import O4_Version

overpass_servers = {
    "DE": "https://overpass-api.de/api/interpreter",
    "KU": "https://overpass.private.coffee/api/interpreter",
    "RU1": "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "RU2": "https://overpass.openstreetmap.ru/api/interpreter",
    "JP": "https://overpass.osm.jp/api/interpreter"
}
overpass_server_choice = "KU"
max_osm_tentatives = 4
# OSM operations policy requires an identifying User-Agent; the default
# python-requests one gets deprioritized or blocked outright.
overpass_user_agent = (
    "Ortho4XP/" + O4_Version.version
    + " (SegFormer fork; +https://github.com/oscarpilote/Ortho4XP)"
)
max_retry_after_wait = 60
# Developer/testing switch. Existing OSM cache is still reused first.
disable_osm_downloads = False
_simulator_water_geometry_cache = {}

################################################################################
class OSM_layer:
    def __init__(self):
        self.dicosmn = (
            {}
        )  # keys are ints (ids) and values are tuple of (lat,lon)
        self.dicosmn_reverse = {}  # reverese of the previous one
        self.dicosmw = {}
        self.next_node_id = -1
        self.next_way_id = -1
        self.next_rel_id = -1
        # rels already sorted out and containing nodeids rather than wayids
        self.dicosmr = {}
        # original rels containing wayids only, not sorted and/or reversed
        self.dicosmrorig = {}
        # ids of objects directly queried, not of child or
        # parent objects pulled indirectly by queries. Since
        # osm ids are only unique per object type we need one for each:
        self.dicosmfirst = {"n": set(), "w": set(), "r": set()}
        self.dicosmtags = {"n": {}, "w": {}, "r": {}}
        self.dicosm = [
            self.dicosmn,
            self.dicosmw,
            self.dicosmr,
            self.dicosmrorig,
            self.dicosmfirst,
            self.dicosmtags,
        ]

    def update_dicosm(self, osm_input, input_tags=None, target_tags=None):
        # input_tags (dict or None) are the input query tags (per osm type)
        # target_tags (dict or None) are the the tags which should be kept 
        # (per osm type) It is expected that if not None the target_tags 
        # contains the input_tags
        initnodes = len(self.dicosmn)
        initways = len(self.dicosmfirst["w"])
        initrels = len(self.dicosmfirst["r"])
        dicosmn_id_map = {}
        dicosmw_id_map = {}
        # osm_input may either refer to an osm filename (e.g. cached data) or
        # to a xml bytestring (direct download)
        if isinstance(osm_input, str):
            osm_file_name = osm_input
            try:
                if osm_file_name[-4:] == ".bz2":
                    pfile = bz2.open(osm_file_name, "rt", encoding="utf-8")
                else:
                    pfile = open(osm_file_name, "r", encoding="utf-8")
            except:
                UI.vprint(
                    1,
                    "    Could not open",
                    osm_file_name,
                    "for reading (corrupted ?).",
                )
                return 0
        elif isinstance(osm_input, bytes):
            pfile = io.StringIO(osm_input.decode(encoding="utf-8"))
        first_line = pfile.readline()
        if "<osm " not in first_line:
            first_line = pfile.readline()
        separator = "'" if "'" in first_line else '"'
        normal_exit = False
        for line in pfile:
            items = line.split(separator)
            if "<node id=" in items[0]:
                osmtype = "n"
                osmid = items[1]
                for j in range(0, len(items)):
                    if items[j] == " lat=":
                        latp = float(items[j + 1])
                    elif items[j] == " lon=":
                        lonp = float(items[j + 1])
                if (lonp, latp) in self.dicosmn_reverse:
                    true_osmid = self.dicosmn_reverse[(lonp, latp)]
                    dicosmn_id_map[osmid] = true_osmid
                    osmid = true_osmid
                else:
                    true_osmid = self.next_node_id
                    dicosmn_id_map[osmid] = true_osmid
                    osmid = true_osmid
                    self.dicosmn_reverse[(lonp, latp)] = osmid
                    self.dicosmn[osmid] = (lonp, latp)
                    self.next_node_id -= 1
            elif "<way id=" in items[0]:
                osmtype = "w"
                osmid = items[1]
                true_osmid = self.next_way_id
                self.next_way_id -= 1
                dicosmw_id_map[osmid] = true_osmid
                osmid = true_osmid
                self.dicosmw[osmid] = []
                if not input_tags:
                    self.dicosmfirst["w"].add(osmid)
            elif "<nd ref=" in items[0]:
                self.dicosmw[osmid].append(dicosmn_id_map[items[1]])
            elif "<relation id=" in items[0]:
                osmtype = "r"
                osmid = items[1]
                true_osmid = self.next_rel_id
                self.next_rel_id -= 1
                osmid = true_osmid
                self.dicosmr[osmid] = {"outer": [], "inner": []}
                self.dicosmrorig[osmid] = {"outer": [], "inner": []}
                dico_rel_check = {"inner": {}, "outer": {}}
                if not input_tags:
                    self.dicosmfirst["r"].add(osmid)
            elif "<member type=" in items[0]:
                role = items[5]
                if items[1] != "way" or role not in ("outer", "inner"):
                    if items[1] == "node":
                        continue  # not necessary to report these
                    UI.lvprint(
                        2,
                        "Relation id=",
                        osmid,
                        "contains a member of type",
                        "'" + items[1] + "'",
                        "and role",
                        "'" + role + "'",
                        "which was not treated (only deal with 'ways' of role ",
                        "'inner' or 'outer').",
                    )
                    continue
                try:
                    wayid = dicosmw_id_map[items[3]]
                except:
                    continue
                self.dicosmrorig[osmid][role].append(wayid)
                endpt1 = self.dicosmw[wayid][0]
                endpt2 = self.dicosmw[wayid][-1]
                if endpt1 == endpt2:
                    self.dicosmr[osmid][role].append(self.dicosmw[wayid])
                else:
                    if endpt1 in dico_rel_check[role]:
                        dico_rel_check[role][endpt1].append(wayid)
                    else:
                        dico_rel_check[role][endpt1] = [wayid]
                    if endpt2 in dico_rel_check[role]:
                        dico_rel_check[role][endpt2].append(wayid)
                    else:
                        dico_rel_check[role][endpt2] = [wayid]
            elif "<tag k=" in items[0]:
                # Do we need to catch that tag ?
                if (
                    (not input_tags)
                    or (("all", "") in target_tags[osmtype])
                    or ((items[1], "") in target_tags[osmtype])
                    or ((items[1], items[3]) in target_tags[osmtype])
                ):
                    if osmid not in self.dicosmtags[osmtype]:
                        self.dicosmtags[osmtype][osmid] = {items[1]: items[3]}
                    else:
                        self.dicosmtags[osmtype][osmid][items[1]] = items[3]
                    # If so, do we need to declare this osmid as a first catch, 
                    # not one only brought with as a child
                    if input_tags and (
                        ((items[1], "") in input_tags[osmtype])
                        or ((items[1], items[3]) in input_tags[osmtype])
                    ):
                        self.dicosmfirst[osmtype].add(osmid)
            elif "</way" in items[0]:
                if not self.dicosmw[osmid]:
                    del self.dicosmw[osmid]
                    self.next_way_id += 1
                    if osmid in self.dicosmfirst["w"]:
                        self.dicosmfirst["w"].remove(osmid)
                    if osmid in self.dicosmtags["w"]:
                        del self.dicosmtags[osmtype][osmid]
            elif "</relation>" in items[0]:
                bad_rel = False
                for role, endpt in (
                    (r, e)
                    for r in ["outer", "inner"]
                    for e in dico_rel_check[r]
                ):
                    if len(dico_rel_check[role][endpt]) != 2:
                        bad_rel = True
                        break
                if bad_rel == True:
                    UI.lvprint(
                        2,
                        "Relation id=",
                        osmid,
                        "is ill formed and was not treated.",
                    )
                    del self.dicosmr[osmid]
                    del self.dicosmrorig[osmid]
                    del dico_rel_check
                    self.next_rel_id += 1
                    if osmid in self.dicosmfirst["r"]:
                        self.dicosmfirst["r"].remove(osmid)
                    if osmid in self.dicosmtags["r"]:
                        del self.dicosmtags["r"][osmid]
                    continue
                for role in ["outer", "inner"]:
                    while dico_rel_check[role]:
                        nodeids = []
                        endpt = next(iter(dico_rel_check[role]))
                        wayid = dico_rel_check[role][endpt][0]
                        endptinit = self.dicosmw[wayid][0]
                        endpt1 = endptinit
                        endpt2 = self.dicosmw[wayid][-1]
                        for nodeid in self.dicosmw[wayid][:-1]:
                            nodeids.append(nodeid)
                        while endpt2 != endptinit:
                            if dico_rel_check[role][endpt2][0] == wayid:
                                wayid = dico_rel_check[role][endpt2][1]
                            else:
                                wayid = dico_rel_check[role][endpt2][0]
                            endpt1 = endpt2
                            if self.dicosmw[wayid][0] == endpt1:
                                endpt2 = self.dicosmw[wayid][-1]
                                for nodeid in self.dicosmw[wayid][:-1]:
                                    nodeids.append(nodeid)
                            else:
                                endpt2 = self.dicosmw[wayid][0]
                                for nodeid in self.dicosmw[wayid][-1:0:-1]:
                                    nodeids.append(nodeid)
                            del dico_rel_check[role][endpt1]
                        nodeids.append(endptinit)
                        self.dicosmr[osmid][role].append(nodeids)
                        del dico_rel_check[role][endptinit]
                if target_tags == None:
                    for wayid in (
                        self.dicosmrorig[osmid]["outer"]
                        + self.dicosmrorig[osmid]["inner"]
                    ):
                        try:
                            self.dicosmfirst["w"].remove(wayid)
                        except:
                            pass
                if not self.dicosmr[osmid]["outer"]:
                    del self.dicosmr[osmid]
                    del self.dicosmrorig[osmid]
                    self.next_rel_id += 1
                    if osmid in self.dicosmfirst["r"]:
                        self.dicosmfirst["r"].remove(osmid)
                    if osmid in self.dicosmtags["r"]:
                        del self.dicosmtags["r"][osmid]
                del dico_rel_check
            elif "</osm>" in items[0]:
                normal_exit = True
        pfile.close()
        if not normal_exit:
            UI.lvprint(
                0,
                "ERROR: OSM overpass server answer was corrupted ",
                "(no ending </OSM> tag)",
            )
            return 0
        UI.vprint(
            2,
            "      A total of "
            + str(len(self.dicosmn) - initnodes)
            + " new node(s), "
            + str(len(self.dicosmfirst["w"]) - initways)
            + " new ways and "
            + str(len(self.dicosmfirst["r"]) - initrels)
            + " new relation(s).",
        )
        return 1

    def write_to_file(self, filename):
        try:
            if filename[-4:] == ".bz2":
                fout = bz2.open(filename, "wt", encoding="utf-8")
            else:
                fout = open(filename, "w", encoding="utf-8")
        except:
            UI.vprint(1, "    Could not open", filename, "for writing.")
            return 0
        fout.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" ' + 
            'generator="Ortho4XP">\n'
        )
        if not len(self.dicosmfirst["n"]):
            for nodeid, (lonp, latp) in self.dicosmn.items():
                fout.write(
                    '  <node id="'
                    + str(nodeid)
                    + '" lat="'
                    + "{:.7f}".format(latp)
                    + '" lon="'
                    + "{:.7f}".format(lonp)
                    + '" version="1"/>\n'
                )
        else:
            for nodeid, (lonp, latp) in self.dicosmn.items():
                if nodeid not in self.dicosmtags["n"]:
                    fout.write(
                        '  <node id="'
                        + str(nodeid)
                        + '" lat="'
                        + "{:.7f}".format(latp)
                        + '" lon="'
                        + "{:.7f}".format(lonp)
                        + '" version="1"/>\n'
                    )
                else:
                    fout.write(
                        '  <node id="'
                        + str(nodeid)
                        + '" lat="'
                        + "{:.7f}".format(latp)
                        + '" lon="'
                        + "{:.7f}".format(lonp)
                        + '" version="1">\n'
                    )
                    for tag in self.dicosmtags["n"][nodeid]:
                        fout.write(
                            '    <tag k="'
                            + tag
                            + '" v="'
                            + self.dicosmtags["n"][nodeid][tag]
                            + '"/>\n'
                        )
                    fout.write("  </node>\n")
        for wayid in tuple(self.dicosmfirst["w"]) + tuple(
            set(self.dicosmw).difference(self.dicosmfirst["w"])
        ):
            fout.write('  <way id="' + str(wayid) + '" version="1">\n')
            for nodeid in self.dicosmw[wayid]:
                fout.write('    <nd ref="' + str(nodeid) + '"/>\n')
            for tag in (
                self.dicosmtags["w"][wayid]
                if wayid in self.dicosmtags["w"]
                else []
            ):
                fout.write(
                    '    <tag k="'
                    + tag
                    + '" v="'
                    + self.dicosmtags["w"][wayid][tag]
                    + '"/>\n'
                )
            fout.write("  </way>\n")
        for relid in tuple(self.dicosmfirst["r"]) + tuple(
            set(self.dicosmrorig).difference(self.dicosmfirst["r"])
        ):
            fout.write('  <relation id="' + str(relid) + '" version="1">\n')
            for wayid in self.dicosmrorig[relid]["outer"]:
                fout.write(
                    '    <member type="way" ref="'
                    + str(wayid)
                    + '" role="outer"/>\n'
                )
            for wayid in self.dicosmrorig[relid]["inner"]:
                fout.write(
                    '    <member type="way" ref="'
                    + str(wayid)
                    + '" role="inner"/>\n'
                )
            for tag in (
                self.dicosmtags["r"][relid]
                if relid in self.dicosmtags["r"]
                else []
            ):
                fout.write(
                    '    <tag k="'
                    + tag
                    + '" v="'
                    + self.dicosmtags["r"][relid][tag]
                    + '"/>\n'
                )
            fout.write("  </relation>\n")
        fout.write("</osm>")
        fout.close()
        return 1

################################################################################
def _create_synthetic_way(osm_layer, lon_lat_points, tags=None, first=True):
    nodeids = []
    for lonp, latp in lon_lat_points:
        node_key = (lonp, latp)
        if node_key in osm_layer.dicosmn_reverse:
            nodeid = osm_layer.dicosmn_reverse[node_key]
        else:
            nodeid = osm_layer.next_node_id
            osm_layer.next_node_id -= 1
            osm_layer.dicosmn_reverse[node_key] = nodeid
            osm_layer.dicosmn[nodeid] = node_key
        nodeids.append(nodeid)
    if len(nodeids) < 2:
        return None
    wayid = osm_layer.next_way_id
    osm_layer.next_way_id -= 1
    osm_layer.dicosmw[wayid] = nodeids
    if first:
        osm_layer.dicosmfirst["w"].add(wayid)
    if tags:
        osm_layer.dicosmtags["w"][wayid] = dict(tags)
    return wayid


def _add_synthetic_way(osm_layer, lon_lat_points, tags):
    """Append one generated way to an OSM layer without writing a cache file."""
    return 1 if _create_synthetic_way(osm_layer, lon_lat_points, tags) else 0


def _add_synthetic_node(osm_layer, lonp, latp, tags):
    node_key = (lonp, latp)
    if node_key in osm_layer.dicosmn_reverse:
        nodeid = osm_layer.dicosmn_reverse[node_key]
    else:
        nodeid = osm_layer.next_node_id
        osm_layer.next_node_id -= 1
        osm_layer.dicosmn_reverse[node_key] = nodeid
        osm_layer.dicosmn[nodeid] = node_key
    osm_layer.dicosmfirst["n"].add(nodeid)
    osm_layer.dicosmtags["n"][nodeid] = dict(tags)
    return nodeid


def _add_synthetic_multipolygon(osm_layer, polygon, tags):
    relation_id = osm_layer.next_rel_id
    osm_layer.next_rel_id -= 1
    osm_layer.dicosmr[relation_id] = {"outer": [], "inner": []}
    osm_layer.dicosmrorig[relation_id] = {"outer": [], "inner": []}
    outer_way_id = _create_synthetic_way(
        osm_layer, list(polygon.exterior.coords), first=False
    )
    if not outer_way_id:
        return 0
    osm_layer.dicosmr[relation_id]["outer"].append(osm_layer.dicosmw[outer_way_id])
    osm_layer.dicosmrorig[relation_id]["outer"].append(outer_way_id)
    for interior in polygon.interiors:
        inner_way_id = _create_synthetic_way(
            osm_layer, list(interior.coords), first=False
        )
        if not inner_way_id:
            continue
        osm_layer.dicosmr[relation_id]["inner"].append(osm_layer.dicosmw[inner_way_id])
        osm_layer.dicosmrorig[relation_id]["inner"].append(inner_way_id)
    osm_layer.dicosmfirst["r"].add(relation_id)
    osm_layer.dicosmtags["r"][relation_id] = dict(tags)
    return 1


def _candidate_xplane_roots():
    roots = []
    try:
        import O4_Overlay_Utils as OVL
    except Exception:
        OVL = None
    try:
        import O4_Config_Utils as CFG
    except Exception:
        CFG = None
    candidates = []
    if OVL:
        candidates.extend((OVL.custom_overlay_src, OVL.custom_overlay_src_alternate))
    if CFG:
        candidates.append(getattr(CFG, "custom_scenery_dir", ""))
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if os.path.basename(candidate).lower() == "custom scenery":
            roots.append(os.path.dirname(candidate))
        if os.path.isdir(os.path.join(candidate, "Custom Scenery")):
            roots.append(candidate)
        if os.path.isdir(os.path.join(candidate, "Global Scenery")):
            roots.append(candidate)
        probe = candidate
        for _ in range(6):
            apt_dat = os.path.join(
                probe,
                "Resources",
                "default scenery",
                "default apt dat",
                "Earth nav data",
                "apt.dat",
            )
            if os.path.isfile(apt_dat):
                roots.append(probe)
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
    unique_roots = []
    seen = set()
    for root in roots:
        real_root = os.path.realpath(root)
        if real_root in seen:
            continue
        seen.add(real_root)
        unique_roots.append(root)
    return unique_roots


def _configured_scenery_source_dirs():
    sources = []
    try:
        import O4_Overlay_Utils as OVL
    except Exception:
        OVL = None
    if OVL:
        sources.extend((OVL.custom_overlay_src, OVL.custom_overlay_src_alternate))
    for root in _candidate_xplane_roots():
        global_scenery = os.path.join(root, "Global Scenery")
        if not os.path.isdir(global_scenery):
            continue
        sources.extend(
            os.path.join(global_scenery, entry.name)
            for entry in os.scandir(global_scenery)
            if entry.is_dir()
        )

    normalized_sources = []
    seen = set()
    for source in sources:
        if not source:
            continue
        source = os.path.abspath(source)
        candidates = [source]
        if os.path.basename(source).lower() == "earth nav data":
            candidates.insert(0, os.path.dirname(source))
        global_scenery = os.path.join(source, "Global Scenery")
        if os.path.isdir(global_scenery):
            candidates.extend(
                os.path.join(global_scenery, entry.name)
                for entry in os.scandir(global_scenery)
                if entry.is_dir()
            )
        for candidate in candidates:
            if not os.path.isdir(os.path.join(candidate, "Earth nav data")):
                continue
            real_path = os.path.realpath(candidate)
            if real_path in seen:
                continue
            seen.add(real_path)
            normalized_sources.append(candidate)
    return normalized_sources


def _configured_overlay_dsf(lat, lon):
    try:
        import O4_Overlay_Utils as OVL
    except Exception:
        return (None, None)
    checked = []
    for overlay_src in _configured_scenery_source_dirs():
        dsf_path = os.path.join(
            overlay_src,
            "Earth nav data",
            FNAMES.long_latlon(lat, lon) + ".dsf",
        )
        checked.append(dsf_path)
        if os.path.exists(dsf_path):
            return (dsf_path, OVL)
    if checked:
        UI.vprint(
            1,
            "      Simulator fallback checked",
            len(checked),
            "DSF path(s); no DSF found for",
            FNAMES.short_latlon(lat, lon) + ".",
        )
    return (None, OVL)


def _candidate_apt_dat_files():
    apt_dat_files = []
    for root in _candidate_xplane_roots():
        apt_dat = os.path.join(
            root,
            "Resources",
            "default scenery",
            "default apt dat",
            "Earth nav data",
            "apt.dat",
        )
        if os.path.isfile(apt_dat):
            apt_dat_files.append(apt_dat)
        custom_scenery = os.path.join(root, "Custom Scenery")
        if os.path.isdir(custom_scenery):
            for entry in sorted(os.scandir(custom_scenery), key=lambda item: item.name.lower()):
                if not entry.is_dir():
                    continue
                custom_apt = os.path.join(entry.path, "Earth nav data", "apt.dat")
                if os.path.isfile(custom_apt):
                    apt_dat_files.insert(0, custom_apt)
    unique_files = []
    seen = set()
    for apt_dat in apt_dat_files:
        real_path = os.path.realpath(apt_dat)
        if real_path in seen:
            continue
        seen.add(real_path)
        unique_files.append(apt_dat)
    return unique_files


def _point_in_tile(latp, lonp, lat, lon, margin=0.05):
    return (
        lat - margin <= latp <= lat + 1 + margin
        and lon - margin <= lonp <= lon + 1 + margin
    )


def _apt_dat_airport_fallback(osm_layer, lat, lon, cached_suffix):
    if cached_suffix != "airports":
        return 0
    apt_dat_files = _candidate_apt_dat_files()
    if not apt_dat_files:
        UI.vprint(1, "      apt.dat airport fallback skipped: no apt.dat found.")
        return 0

    airport_count = 0
    runway_count = 0
    seen_airports = set()
    for apt_dat in apt_dat_files:
        try:
            with open(apt_dat, "r", encoding="utf-8", errors="ignore") as apt_file:
                current_airport = None
                current_has_runway = False
                pending_runways = []
                for raw_line in apt_file:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    code = parts[0]
                    if code in ("1", "16", "17"):
                        if current_airport and current_has_runway:
                            apt_key = current_airport["id"]
                            if apt_key not in seen_airports:
                                seen_airports.add(apt_key)
                                _add_synthetic_node(
                                    osm_layer,
                                    current_airport["lon"],
                                    current_airport["lat"],
                                    current_airport["tags"],
                                )
                                airport_count += 1
                                for runway in pending_runways:
                                    runway_count += _add_synthetic_way(
                                        osm_layer, runway["points"], runway["tags"]
                                    )
                        current_airport = None
                        current_has_runway = False
                        pending_runways = []
                        if len(parts) < 5:
                            continue
                        airport_id = parts[4]
                        name = " ".join(parts[5:]) if len(parts) > 5 else airport_id
                        tags = {
                            "aeroway": "aerodrome",
                            "local_ref": airport_id,
                            "name": name,
                            "source": "xplane_apt_dat",
                        }
                        if len(airport_id) == 4:
                            tags["icao"] = airport_id
                        current_airport = {
                            "id": airport_id,
                            "name": name,
                            "lat": None,
                            "lon": None,
                            "tags": tags,
                        }
                    elif code == "100" and current_airport and len(parts) >= 20:
                        try:
                            width_m = float(parts[1])
                            lat1 = float(parts[9])
                            lon1 = float(parts[10])
                            lat2 = float(parts[18])
                            lon2 = float(parts[19])
                        except (ValueError, IndexError):
                            continue
                        center_lat = (lat1 + lat2) / 2
                        center_lon = (lon1 + lon2) / 2
                        if not (
                            _point_in_tile(lat1, lon1, lat, lon)
                            or _point_in_tile(lat2, lon2, lat, lon)
                            or _point_in_tile(center_lat, center_lon, lat, lon)
                        ):
                            continue
                        current_airport["lat"] = (
                            center_lat
                            if current_airport["lat"] is None
                            else (current_airport["lat"] + center_lat) / 2
                        )
                        current_airport["lon"] = (
                            center_lon
                            if current_airport["lon"] is None
                            else (current_airport["lon"] + center_lon) / 2
                        )
                        current_has_runway = True
                        pending_runways.append(
                            {
                                "points": [(lon1, lat1), (lon2, lat2)],
                                "tags": {
                                    "aeroway": "runway",
                                    "width": str(width_m),
                                    "source": "xplane_apt_dat",
                                },
                            }
                        )
                if current_airport and current_has_runway:
                    apt_key = current_airport["id"]
                    if apt_key not in seen_airports:
                        seen_airports.add(apt_key)
                        _add_synthetic_node(
                            osm_layer,
                            current_airport["lon"],
                            current_airport["lat"],
                            current_airport["tags"],
                        )
                        airport_count += 1
                        for runway in pending_runways:
                            runway_count += _add_synthetic_way(
                                osm_layer, runway["points"], runway["tags"]
                            )
        except Exception as exc:
            UI.vprint(1, "      apt.dat airport fallback failed:", apt_dat, exc)

    if airport_count:
        UI.vprint(
            1,
            "      apt.dat airport fallback added",
            airport_count,
            "airport(s) and",
            runway_count,
            "runway(s).",
        )
        return 1
    UI.vprint(1, "      apt.dat airport fallback found no airports in tile.")
    return 0


def _terrain_def_is_water(terrain_def, dsf_path=None):
    normalized = terrain_def.replace("\\", "/").strip()
    stem = os.path.splitext(os.path.basename(normalized))[0].lower()
    if stem in ("water", "terrain_water"):
        return True
    if not normalized.lower().endswith(".ter") or not dsf_path:
        return False

    dsf_parts = os.path.normpath(dsf_path).split(os.sep)
    try:
        end_nav_idx = dsf_parts.index("Earth nav data")
    except ValueError:
        return False
    scenery_root = os.sep.join(dsf_parts[:end_nav_idx])
    ter_path = os.path.join(scenery_root, *normalized.split("/"))
    if not os.path.isfile(ter_path):
        return False
    try:
        with open(ter_path, "r", encoding="utf-8", errors="ignore") as ter_file:
            return any("WATER_COLOR_MASK" in line for line in ter_file)
    except Exception:
        return False


def _triangles_from_primitive(primitive_type, vertices):
    if len(vertices) < 3:
        return []
    if primitive_type == 0:
        return [vertices[i : i + 3] for i in range(0, len(vertices) - 2, 3)]
    if primitive_type == 1:
        return [vertices[i : i + 3] for i in range(len(vertices) - 2)]
    if primitive_type == 2:
        return [[vertices[0], vertices[i], vertices[i + 1]] for i in range(1, len(vertices) - 1)]
    return []


def _as_multipolygon(geom):
    if geom.is_empty:
        return geometry.MultiPolygon()
    if geom.geom_type == "Polygon":
        return geometry.MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    if "Collection" in geom.geom_type:
        return geometry.MultiPolygon([item for item in geom.geoms if item.geom_type == "Polygon"])
    return geometry.MultiPolygon()


def _water_geometry_from_dsf_text(lines, lat, lon, dsf_path=None):
    terrain_defs = []
    water_triangles = []
    current_patch_is_water = False
    current_primitive_type = None
    current_vertices = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("TERRAIN_DEF "):
            terrain_defs.append(line.split(None, 1)[1])
        elif line.startswith("BEGIN_PATCH "):
            parts = line.split()
            current_patch_is_water = False
            try:
                terrain_idx = int(parts[1])
                flags = int(parts[4])
            except (IndexError, ValueError):
                continue
            if terrain_idx < len(terrain_defs):
                current_patch_is_water = (
                    _terrain_def_is_water(terrain_defs[terrain_idx], dsf_path)
                    and bool(flags & 1)
                )
        elif line.startswith("BEGIN_PRIMITIVE "):
            try:
                current_primitive_type = int(line.split()[1])
            except (IndexError, ValueError):
                current_primitive_type = None
            current_vertices = []
        elif line.startswith("PATCH_VERTEX ") and current_patch_is_water:
            parts = line.split()
            try:
                current_vertices.append((float(parts[1]), float(parts[2])))
            except (IndexError, ValueError):
                pass
        elif line.startswith("END_PRIMITIVE"):
            if current_patch_is_water and current_primitive_type is not None:
                water_triangles.extend(
                    _triangles_from_primitive(current_primitive_type, current_vertices)
                )
            current_primitive_type = None
            current_vertices = []
        elif line.startswith("END_PATCH"):
            current_patch_is_water = False

    polygons = []
    tile_bounds = geometry.box(lon, lat, lon + 1, lat + 1)
    for triangle in water_triangles:
        try:
            polygon = geometry.Polygon(triangle)
        except Exception:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.area:
            continue
        clipped = polygon.intersection(tile_bounds)
        if clipped.is_empty:
            continue
        polygons.extend(_as_multipolygon(clipped).geoms)
    if not polygons:
        return geometry.MultiPolygon()
    return _as_multipolygon(ops.unary_union(polygons).buffer(0))


def _read_simulator_dsf_text(dsf_path, ovl_module, lat, lon, tmp_suffix):
    import shutil
    import subprocess

    dsftool = ovl_module.dsftool_cmd.strip()
    if not dsftool or not os.path.exists(dsftool):
        UI.vprint(1, "      Simulator fallback skipped: DSFTool not found.")
        return []
    tmp_dsf = os.path.join(
        FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + "_" + tmp_suffix + ".dsf"
    )
    tmp_txt = os.path.join(
        FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + "_" + tmp_suffix + ".txt"
    )
    tmp_extract_dir = os.path.join(
        FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + "_" + tmp_suffix
    )
    try:
        os.makedirs(FNAMES.Tmp_dir, exist_ok=True)
        shutil.copyfile(dsf_path, tmp_dsf)
        with open(tmp_dsf, "rb") as dsf_file:
            dsfid = dsf_file.read(2).decode("ascii", errors="ignore")
        if dsfid == "7z":
            archive_path = tmp_dsf
            os.makedirs(tmp_extract_dir, exist_ok=True)
            subprocess.run(
                [
                    ovl_module.unzip_cmd,
                    "e",
                    "-y",
                    f"-o{tmp_extract_dir}",
                    archive_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=getattr(ovl_module, "_CREATE_NO_WINDOW", 0),
                check=True,
            )
            extracted_dsf = os.path.join(
                tmp_extract_dir, FNAMES.short_latlon(lat, lon) + ".dsf"
            )
            if not os.path.isfile(extracted_dsf):
                extracted_dsfs = [
                    os.path.join(tmp_extract_dir, name)
                    for name in os.listdir(tmp_extract_dir)
                    if name.lower().endswith(".dsf")
                ]
                if not extracted_dsfs:
                    raise FileNotFoundError("No DSF was extracted from archive")
                extracted_dsf = extracted_dsfs[0]
            os.remove(archive_path)
            shutil.move(extracted_dsf, tmp_dsf)
        subprocess.run(
            [dsftool, "-dsf2text", tmp_dsf, tmp_txt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=getattr(ovl_module, "_CREATE_NO_WINDOW", 0),
            check=True,
        )
        with open(tmp_txt, "r", encoding="utf-8", errors="ignore") as text_file:
            return list(text_file)
    finally:
        for tmp_path in (tmp_dsf, tmp_txt, tmp_dsf + ".7z"):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        tmp_txt_prefix = os.path.basename(tmp_txt) + "."
        try:
            for name in os.listdir(FNAMES.Tmp_dir):
                if name.startswith(tmp_txt_prefix):
                    os.remove(os.path.join(FNAMES.Tmp_dir, name))
        except Exception:
            pass
        try:
            if os.path.isdir(tmp_extract_dir):
                shutil.rmtree(tmp_extract_dir)
        except Exception:
            pass


def _parse_simulator_water_dsf(dsf_path, ovl_module, lat, lon):
    try:
        cache_key = (
            os.path.realpath(dsf_path),
            os.path.getmtime(dsf_path),
            os.path.getsize(dsf_path),
            lat,
            lon,
        )
        if cache_key in _simulator_water_geometry_cache:
            return _simulator_water_geometry_cache[cache_key]
        lines = _read_simulator_dsf_text(
            dsf_path, ovl_module, lat, lon, "fallback_water_dsf"
        )
        water_area = _water_geometry_from_dsf_text(lines, lat, lon, dsf_path)
        _simulator_water_geometry_cache[cache_key] = water_area
        return water_area
    except Exception as exc:
        UI.vprint(1, "      Simulator water fallback failed:", exc)
        return geometry.MultiPolygon()


def _on_tile_boundary(point, lat, lon, eps=1e-7):
    x, y = point
    return (
        abs(x - lon) <= eps
        or abs(x - (lon + 1)) <= eps
        or abs(y - lat) <= eps
        or abs(y - (lat + 1)) <= eps
    )


def _tile_boundary_segment(p1, p2, lat, lon, eps=1e-7):
    return (
        (abs(p1[0] - p2[0]) <= eps and (abs(p1[0] - lon) <= eps or abs(p1[0] - (lon + 1)) <= eps))
        or (abs(p1[1] - p2[1]) <= eps and (abs(p1[1] - lat) <= eps or abs(p1[1] - (lat + 1)) <= eps))
    )


def _coastline_segments_from_water_polygon(polygon, lat, lon):
    if not polygon.boundary.intersects(geometry.box(lon, lat, lon + 1, lat + 1).boundary):
        return []

    def exterior_segments(coords):
        coords = list(coords)[::-1]
        segment_count = len(coords) - 1
        if segment_count <= 0:
            return []
        start = 0
        for idx in range(segment_count):
            if _tile_boundary_segment(coords[idx], coords[idx + 1], lat, lon):
                start = (idx + 1) % segment_count
                break
        ordered = coords[start:segment_count] + coords[: start + 1]
        lines = []
        current = []
        for p1, p2 in zip(ordered, ordered[1:]):
            if _tile_boundary_segment(p1, p2, lat, lon):
                if len(current) >= 2:
                    lines.append(current)
                current = []
                continue
            if not current:
                current = [p1]
            current.append(p2)
        if len(current) >= 2:
            lines.append(current)
        return lines

    segments = exterior_segments(polygon.exterior.coords)
    for interior in polygon.interiors:
        ring = list(interior.coords)
        if not geometry.LinearRing(ring).is_ccw:
            ring = ring[::-1]
        segments.append(ring)
    return segments


def _simulator_water_fallback(osm_layer, lat, lon, cached_suffix):
    if cached_suffix not in ("water", "coastline"):
        return 0
    if cached_suffix == "coastline":
        UI.vprint(
            1,
            "      Simulator coastline fallback skipped: default DSF water",
            "is only used for inland water bodies.",
        )
        return 0
    dsf_path, ovl_module = _configured_overlay_dsf(lat, lon)
    if not dsf_path or not ovl_module:
        UI.vprint(
            1,
            "      Simulator water fallback skipped: default scenery source",
            "is not configured or missing.",
        )
        return 0
    UI.vprint(1, "      Trying simulator water fallback from", dsf_path)
    water_area = _parse_simulator_water_dsf(dsf_path, ovl_module, lat, lon)
    if water_area.is_empty:
        UI.vprint(1, "      Simulator water fallback found no water mesh patches.")
        return 0

    count = 0
    tags = {"natural": "water", "source": "xplane_default_scenery"}
    for polygon in water_area.geoms:
        count += _add_synthetic_multipolygon(osm_layer, polygon, tags)
    if count:
        UI.vprint(
            1,
            "      Simulator water fallback added",
            count,
            "water polygon(s).",
        )
        return 1
    UI.vprint(1, "      Simulator water fallback found no usable geometry.")
    return 0


def _parse_simulator_network_dsf(dsf_path, ovl_module, lat, lon):
    import shutil
    import subprocess

    dsftool = ovl_module.dsftool_cmd.strip()
    if not dsftool or not os.path.exists(dsftool):
        UI.vprint(1, "      Simulator fallback skipped: DSFTool not found.")
        return []
    tmp_dsf = os.path.join(FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + ".dsf")
    tmp_txt = os.path.join(
        FNAMES.Tmp_dir, FNAMES.short_latlon(lat, lon) + "_fallback_dsf.txt"
    )
    try:
        os.makedirs(FNAMES.Tmp_dir, exist_ok=True)
        shutil.copy(dsf_path, tmp_dsf)
        with open(tmp_dsf, "rb") as dsf_file:
            dsfid = dsf_file.read(2).decode("ascii", errors="ignore")
        if dsfid == "7z":
            archive_path = tmp_dsf + ".7z"
            os.replace(tmp_dsf, archive_path)
            subprocess.run(
                [
                    ovl_module.unzip_cmd,
                    "e",
                    f"-o{FNAMES.Tmp_dir}",
                    archive_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=getattr(ovl_module, "_CREATE_NO_WINDOW", 0),
                check=True,
            )
            os.remove(archive_path)
        subprocess.run(
            [dsftool, "-dsf2text", tmp_dsf, tmp_txt],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            creationflags=getattr(ovl_module, "_CREATE_NO_WINDOW", 0),
            check=True,
        )
        segments = []
        current_points = None
        with open(tmp_txt, "r", encoding="utf-8", errors="ignore") as text_file:
            for raw_line in text_file:
                line = raw_line.strip()
                if line.startswith("BEGIN_SEGMENT "):
                    parts = line.split()
                    try:
                        current_points = [(float(parts[4]), float(parts[5]))]
                    except (IndexError, ValueError):
                        current_points = None
                elif line.startswith("SHAPE_POINT ") and current_points is not None:
                    parts = line.split()
                    try:
                        current_points.append((float(parts[1]), float(parts[2])))
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("END_SEGMENT ") and current_points is not None:
                    parts = line.split()
                    try:
                        current_points.append((float(parts[2]), float(parts[3])))
                    except (IndexError, ValueError):
                        pass
                    if len(current_points) >= 2:
                        segments.append(current_points)
                    current_points = None
        return segments
    except Exception as exc:
        UI.vprint(1, "      Simulator network fallback failed:", exc)
        return []
    finally:
        for tmp_path in (tmp_dsf, tmp_txt):
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


def _simulator_network_fallback(osm_layer, lat, lon, cached_suffix):
    if cached_suffix not in ("big_roads", "small_roads"):
        return 0
    dsf_path, ovl_module = _configured_overlay_dsf(lat, lon)
    if not dsf_path or not ovl_module:
        UI.vprint(
            1,
            "      Simulator network fallback skipped: default overlay source",
            "is not configured or missing.",
        )
        return 0
    UI.vprint(1, "      Trying simulator network fallback from", dsf_path)
    highway_type = "primary" if cached_suffix == "big_roads" else "residential"
    tags = {"highway": highway_type, "source": "xplane_default_scenery"}
    segment_count = 0
    for segment in _parse_simulator_network_dsf(dsf_path, ovl_module, lat, lon):
        segment_count += _add_synthetic_way(osm_layer, segment, tags)
    if segment_count:
        UI.vprint(
            1,
            "      Simulator network fallback added",
            segment_count,
            "road segment(s).",
        )
        return 1
    UI.vprint(1, "      Simulator network fallback found no usable segments.")
    return 0


def _layer_has_data(osm_layer):
    return bool(
        osm_layer.dicosmfirst["n"]
        or osm_layer.dicosmfirst["w"]
        or osm_layer.dicosmfirst["r"]
    )


def _apply_failed_download_fallback(osm_layer, lat, lon, cached_suffix):
    if _apt_dat_airport_fallback(osm_layer, lat, lon, cached_suffix):
        return 1
    if cached_suffix == "coastline":
        UI.vprint(
            1,
            "      Coastline fallback disabled;",
            "continuing without that OSM layer.",
        )
        return 1
    if _simulator_water_fallback(osm_layer, lat, lon, cached_suffix):
        return 1
    if _simulator_network_fallback(osm_layer, lat, lon, cached_suffix):
        return 1
    if cached_suffix in ("airports", "coastline", "water"):
        UI.vprint(
            1,
            "      No simulator fallback available for",
            cached_suffix + ";",
            "continuing without that OSM layer.",
        )
    return 1 if _layer_has_data(osm_layer) else 0

################################################################################
def OSM_queries_to_OSM_layer(
    queries,
    osm_layer,
    lat,
    lon,
    tags_of_interest=[],
    server_code=None,
    cached_suffix="",
):
    # this one is a bit complicated by a few checks of existing cached data 
    # which had different filenames is versions prior to 1.30
    target_tags = {"n": [], "w": [], "r": []}
    input_tags = {"n": [], "w": [], "r": []}
    for query in queries:
        for tag in [query] if isinstance(query, str) else query:
            items = tag.split('"')
            osm_type = items[0][0]
            try:
                target_tags[osm_type].append((items[1], items[3]))
                input_tags[osm_type].append((items[1], items[3]))
            except:
                target_tags[osm_type].append((items[1], ""))
                input_tags[osm_type].append((items[1], ""))
            for tag in tags_of_interest:
                if isinstance(tag, str):
                    if (tag, "") not in target_tags[osm_type]:
                        target_tags[osm_type].append((tag, ""))
                else:
                    if tag not in target_tags[osm_type]:
                        target_tags[osm_type].append(tag)
    cached_data_filename = FNAMES.osm_cached(lat, lon, cached_suffix)
    if cached_suffix and os.path.isfile(cached_data_filename):
        UI.vprint(1, "    * Recycling OSM data from", cached_data_filename)
        return osm_layer.update_dicosm(
            cached_data_filename, input_tags, target_tags
        )
    if cached_suffix and PBF.pbf_ready():
        sliced_filename = PBF.slice_tile_layer(lat, lon, cached_suffix)
        if sliced_filename:
            UI.vprint(
                1,
                "    * Slicing OSM data from the local planet extract:",
                sliced_filename,
            )
            return osm_layer.update_dicosm(
                sliced_filename, input_tags, target_tags
            )
        UI.vprint(
            1,
            "    * Local planet slicing failed, falling back to Overpass.",
        )
    for query in queries:
        # look first for cached data (old scheme)
        if isinstance(query, str):
            old_cached_data_filename = FNAMES.osm_old_cached(lat, lon, query)
            if os.path.isfile(old_cached_data_filename):
                UI.vprint(1, "    * Recycling OSM data for", query)
                osm_layer.update_dicosm(
                    old_cached_data_filename, input_tags, target_tags
                )
                continue
        UI.vprint(1, "    * Downloading OSM data for", query)
        response = get_overpass_data(
            query, (lat, lon, lat + 1, lon + 1), server_code
        )
        if UI.red_flag:
            return 0
        if not response:
            UI.logprint(
                "No valid answer for",
                query,
                "after",
                max_osm_tentatives,
                ", skipping it.",
            )
            UI.vprint(
                1,
                "      No valid answer after",
                max_osm_tentatives,
                ", skipping it.",
            )
            return _apply_failed_download_fallback(
                osm_layer, lat, lon, cached_suffix
            )
        osm_layer.update_dicosm(response, input_tags, target_tags)
    if cached_suffix:
        osm_layer.write_to_file(cached_data_filename)
    return 1

################################################################################
def OSM_query_to_OSM_layer(
    query,
    bbox,
    osm_layer,
    tags_of_interest=[],
    server_code=None,
    cached_file_name="",
):
    # this one is simpler and does not depend on the notion of tile
    target_tags = {"n": [], "w": [], "r": []}
    input_tags = {"n": [], "w": [], "r": []}
    for tag in [query] if isinstance(query, str) else query:
        items = tag.split('"')
        osm_type = items[0][0]
        try:
            target_tags[osm_type].append((items[1], items[3]))
            input_tags[osm_type].append((items[1], items[3]))
        except:
            target_tags[osm_type].append((items[1], ""))
            input_tags[osm_type].append((items[1], ""))
        for tag in tags_of_interest:
            if isinstance(tag, str):
                target_tags[osm_type].append((tag, ""))
            else:
                target_tags[osm_type].append(tag)
    if cached_file_name and os.path.isfile(cached_file_name):
        UI.vprint(1, "    * Recycling OSM data from", cached_file_name)
        osm_layer.update_dicosm(cached_file_name, input_tags, target_tags)
    else:
        response = get_overpass_data(query, bbox, server_code)
        if UI.red_flag:
            return 0
        if not response:
            UI.lvprint(
                1,
                "      No valid answer for",
                query,
                "after",
                max_osm_tentatives,
                ", skipping it.",
            )
            return 0
        osm_layer.update_dicosm(response, input_tags, target_tags)
        if cached_file_name:
            osm_layer.write_to_file(cached_file_name)
    return 1

################################################################################
def get_overpass_data(query, bbox, server_code=None):
    if disable_osm_downloads:
        UI.vprint(
            1,
            "        OSM downloads disabled for fallback testing; skipping live request.",
        )
        return 0
    server_codes = list(overpass_servers.keys())
    if server_code in server_codes:
        start_idx = server_codes.index(server_code)
    elif overpass_server_choice == "random":
        start_idx = random.randrange(len(server_codes))
    elif overpass_server_choice in server_codes:
        start_idx = server_codes.index(overpass_server_choice)
    else:
        start_idx = 0
    if isinstance(query, str):
        overpass_query = query + str(bbox) + ";"
    else:  # query is a tuple
        overpass_query = "".join([x + str(bbox) + ";" for x in query])
    payload = "(" + overpass_query + ");(._;>>;);out meta;"
    tentative = 1
    while True:
        # Rotate through the server pool on every failed tentative so one
        # busy or blocking server cannot exhaust all retries on its own.
        true_server_code = server_codes[
            (start_idx + tentative - 1) % len(server_codes)
        ]
        base_url = overpass_servers[true_server_code]
        wait = 2 ** tentative
        UI.vprint(3, base_url, payload)
        try:
            s = requests.Session()
            r = s.post(
                base_url,
                data={"data": payload},
                headers={"User-Agent": overpass_user_agent},
                timeout=60,
            )
            UI.vprint(3, "OSM response status :", r.status_code)
            if r.status_code == 200:
                if (
                    b"</osm>" not in r.content[-10:]
                    and b"</OSM>" not in r.content[-10:]
                ):
                    UI.vprint(
                        1,
                        "        OSM server",
                        true_server_code,
                        "sent a corrupted answer (no closing </osm> tag in ",
                        "answer), new tentative in",
                        wait,
                        "sec...",
                    )
                elif len(r.content) <= 1000 and b"error" in r.content:
                    UI.vprint(
                        1,
                        "        OSM server",
                        true_server_code,
                        "sent us an error code for the data (data too big ?), ",
                        "new tentative in",
                        wait,
                        "sec...",
                    )
                else:
                    return r.content
            else:
                if r.status_code in (429, 504):
                    try:
                        wait = min(
                            int(r.headers.get("Retry-After") or wait),
                            max_retry_after_wait,
                        )
                    except ValueError:
                        pass
                UI.vprint(
                    1,
                    "        OSM server",
                    true_server_code,
                    "rejected our query (status " + str(r.status_code) + "),",
                    "new tentative in",
                    wait,
                    "sec...",
                )
        except:
            UI.vprint(
                1,
                "        OSM server",
                true_server_code,
                "was too busy or unreachable, new tentative in",
                wait,
                "sec...",
            )
        if tentative >= max_osm_tentatives:
            return 0
        if UI.red_flag:
            return 0
        time.sleep(wait)
        tentative += 1

################################################################################
def OSM_to_MultiLineString(
    osm_layer, lat, lon, tags_for_exclusion=set(), filter=None
):
    multiline = []
    multiline_reject = []
    todo = len(osm_layer.dicosmfirst["w"])
    step = int(todo / 100) + 1
    done = 0
    filtered_segs = 0
    for wayid in osm_layer.dicosmfirst["w"]:
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        if (
            tags_for_exclusion
            and wayid in osm_layer.dicosmtags["w"]
            and not set(osm_layer.dicosmtags["w"][wayid].keys()).isdisjoint(
                tags_for_exclusion
            )
        ):
            done += 1
            continue
        way = numpy.round(
            numpy.array(
                [
                    osm_layer.dicosmn[nodeid]
                    for nodeid in osm_layer.dicosmw[wayid]
                ],
                dtype=numpy.float64,
            )
            - numpy.array([[lon, lat]], dtype=numpy.float64),
            7,
        )
        if filter and not filter(way, filtered_segs):
            try:
                multiline_reject.append(geometry.LineString(way))
            except:
                pass
            done += 1
            continue
        try:
            multiline.append(geometry.LineString(way))
            filtered_segs += len(way)
        except:
            pass
        done += 1
    UI.progress_bar(1, 100)
    if not filter:
        return geometry.MultiLineString(multiline)
    else:
        UI.vprint(2, "      Number of filtered segs :", filtered_segs)
        return (
            geometry.MultiLineString(multiline),
            geometry.MultiLineString(multiline_reject),
        )

################################################################################
def OSM_to_MultiPolygon(osm_layer, lat, lon, filter=None):
    multilist = []
    excludelist = []
    todo = len(osm_layer.dicosmfirst["w"]) + len(osm_layer.dicosmfirst["r"])
    step = int(todo / 100) + 1
    done = 0
    for wayid in osm_layer.dicosmfirst["w"]:
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        if osm_layer.dicosmw[wayid][0] != osm_layer.dicosmw[wayid][-1]:
            UI.logprint(
                "Non closed way starting at",
                osm_layer.dicosmn[osm_layer.dicosmw[wayid][0]],
                ", skipped.",
            )
            done += 1
            continue
        way = numpy.round(
            numpy.array(
                [
                    osm_layer.dicosmn[nodeid]
                    for nodeid in osm_layer.dicosmw[wayid]
                ],
                dtype=numpy.float64,
            )
            - numpy.array([[lon, lat]], dtype=numpy.float64),
            7,
        )
        try:
            pol = geometry.Polygon(way)
            if not pol.area:
                continue
            if not pol.is_valid:
                UI.logprint(
                    "Invalid OSM way starting at",
                    osm_layer.dicosmn[osm_layer.dicosmw[wayid][0]],
                    ", skipped.",
                )
                done += 1
                continue
        except Exception as e:
            UI.vprint(2, e)
            done += 1
            continue
        if filter and filter(pol, wayid, osm_layer.dicosmtags["w"]):
            excludelist.append(pol)
        else:
            multilist.append(pol)
        done += 1
    for relid in osm_layer.dicosmfirst["r"]:
        if done % step == 0:
            UI.progress_bar(1, int(100 * done / todo))
        try:
            multiout = [
                geometry.Polygon(
                    numpy.round(
                        numpy.array(
                            [osm_layer.dicosmn[nodeid] for nodeid in nodelist],
                            dtype=numpy.float64,
                        )
                        - numpy.array([lon, lat], dtype=numpy.float64),
                        7,
                    )
                )
                for nodelist in osm_layer.dicosmr[relid]["outer"]
            ]
            multiout = ops.unary_union(
                [geom for geom in multiout if geom.is_valid]
            )
            multiin = [
                geometry.Polygon(
                    numpy.round(
                        numpy.array(
                            [osm_layer.dicosmn[nodeid] for nodeid in nodelist],
                            dtype=numpy.float64,
                        )
                        - numpy.array([lon, lat], dtype=numpy.float64),
                        7,
                    )
                )
                for nodelist in osm_layer.dicosmr[relid]["inner"]
            ]
            multiin = ops.unary_union(
                [geom for geom in multiin if geom.is_valid]
            )
        except Exception as e:
            UI.logprint(e)
            done += 1
            continue
        multipol = multiout.difference(multiin)
        if filter and filter(multipol, relid, osm_layer.dicosmtags["r"]):
            targetlist = excludelist
        else:
            targetlist = multilist
        for pol in (
            multipol.geoms
            if (
                "Multi" in multipol.geom_type
                or "Collection" in multipol.geom_type
            )
            else [multipol]
        ):
            if not pol.area:
                done += 1
                continue
            if not pol.is_valid:
                UI.logprint(
                    "Relation",
                    relid,
                    "contains an invalid polygon which was discarded",
                )
                done += 1
                continue
            targetlist.append(pol)
        done += 1
    if filter:
        ret_val = (
            geometry.MultiPolygon(multilist),
            geometry.MultiPolygon(excludelist),
        )
        UI.vprint(
            2,
            "    Total number of geometries:",
            len(ret_val[0].geoms),
            len(ret_val[1].geoms),
        )
    else:
        ret_val = geometry.MultiPolygon(multilist)
        UI.vprint(2, "    Total number of geometries:", len(ret_val.geoms))
    UI.progress_bar(1, 100)
    return ret_val
