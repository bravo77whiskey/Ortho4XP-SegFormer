"""Style briefs for the free-design (level 2) facades, per page and scheme.

Each brief names one concrete, recognisable building type, so the model
does not drift into a generic "European" look and the schemes of one page
stay distinct.  The wall colour hints follow the kit's palette in
regions.py loosely; the model is free to deviate.
"""

REGION_LINE = {
    "europe": "a real street in central / northern Europe",
    "asia": "a real street in East Asia (Taiwan, southern China, Japan)",
    "africa": "a real street in sub-Saharan Africa",
    "ind": "an industrial estate (used worldwide)",
    "north_america": "a real suburban / small-town street in the USA or "
                     "Canada",
    "mediterranean": "a real street in southern Europe (Spain, Italy, "
                     "Portugal, Greece)",
    "south_america": "a real street in South America (Brazil, Argentina, "
                     "Peru, Colombia)",
    "australia_oceania": "a real suburban street in Australia or New "
                         "Zealand",
    "generic": "a real ordinary street (no strong regional style)",
}

ROOF_WORDS = {
    "clay_interlock": "interlocking clay roof tiles",
    "concrete_tile": "concrete roof tiles",
    "slate": "natural slate roof",
    "gravel": "flat roof covered in gravel ballast",
    "membrane": "flat roof with a bitumen / PVC membrane",
    "kawara": "East Asian grey glazed kawara clay tiles",
    "standing_seam": "standing-seam metal roof",
    "trapezoid": "trapezoidal profiled steel roof sheets",
    "corrugated": "corrugated iron roof sheets",
    "concrete_slab": "flat concrete roof slab with a parapet-less deck",
    "fibre_cement": "corrugated fibre-cement roof sheets",
    "shingle": "asphalt shingle roof",
    "barrel": "barrel (Spanish) clay tile roof",
}

BRIEFS = {
    "europe_res_a": {
        "s1": "1970s West German detached house (Einfamilienhaus): smooth "
              "off-white render, white PVC tilt-and-turn windows with "
              "external roller-shutter boxes and grey plastic slats (some "
              "half down), narrow stone or concrete sills, a dark grey "
              "plinth, a simple canopy over each front door.",
        "s2": "1950s Bavarian / Austrian house: warm ochre-cream lime render "
              "with painted lighter window surrounds (Faschen), white "
              "wooden casement windows with slim glazing bars, green or "
              "brown wooden front doors, a stone-grey base course.",
        "s3": "1930s Dutch / North German terraced house in dark red "
              "wire-cut brick with recessed mortar, soldier-course brick "
              "lintels and brick sills, white-painted timber windows, "
              "front doors painted deep green, blue or oxblood.",
    },
    "europe_res_b": {
        "s1": "1970s French suburban pavillon: light grey roughcast "
              "(crepi) render, white PVC windows with roller shutters, "
              "plain cement sills, a darker splash zone near the ground.",
        "s2": "Belgian / northern French town house: salmon-pink painted "
              "render, painted stone window surrounds and sills, tall "
              "narrow windows with folding wooden shutters (volets) in a "
              "faded colour, panelled wooden front doors.",
        "s3": "Danish / North German house in yellow brick (gule "
              "mursten) with pale mortar, white-painted windows with a "
              "central mullion, a plain brick band above the ground-floor "
              "windows, dark-painted front doors.",
    },
    "europe_apt": {
        "a1": "1960s German social-housing block: off-white render, "
              "recessed loggias with metal balustrades, flower boxes and "
              "the odd satellite dish, roller-shutter windows, entrance "
              "doors of glass and aluminium with a small canopy.",
        "a2": "East German WBS-70 prefab panel block (Plattenbau): grey "
              "exposed-aggregate concrete panels with visible sealed "
              "joints, large windows, recessed loggias, partly renovated "
              "panels in a lighter tone, plain steel-and-glass entrance "
              "doors.",
        "c1": "Late-19th-century Gruenderzeit mixed-use building: cream "
              "stucco with moulded window surrounds and small pediments, "
              "tall windows, ground floor of separate small shop units, "
              "each with a shop window, door and a sign board with NO "
              "readable text.",
        "c2": "1990s-2000s city commercial building: light grey render, "
              "large aluminium-framed windows, ground floor of glass shop "
              "fronts between narrow piers, sign boards with NO readable "
              "text.",
    },
    "asia_res_a": {
        "s1": "1980s-90s Japanese suburban detached house: pale beige "
              "fibre-cement siding panels with a faint stone/tile imprint, "
              "aluminium sliding windows (some with frosted glass), "
              "an outdoor air-conditioner unit here and there, a simple "
              "metal canopy over the entrance door.",
        "s2": "Taiwanese townhouse (tou-tian-cuo) clad in small cream / "
              "off-white ceramic facade tiles, aluminium sliding windows "
              "behind protruding steel security cages (tie-chuang), split "
              "air-conditioner units on brackets, a steel entrance door.",
        "s3": "Southern Chinese rural concrete house: grey cement render "
              "with a slightly uneven trowel finish, aluminium sliding "
              "windows with thin steel bars, a plain steel front door.",
    },
    "asia_res_b": {
        "s1": "Hong Kong New Territories / Guangdong village house: pale "
              "green painted render, aluminium sliding windows with white "
              "steel security grilles, AC units, a narrow canopy over the "
              "door.",
        "s2": "Taiwanese rural house: cream-yellow painted render over "
              "brick, aluminium sliding windows, frosted glass on small "
              "windows, a red-painted steel entrance door.",
        "s3": "Chinese county-town house: salmon-pink wall tiles on the "
              "upper floors and grey granite-look tiles on the ground "
              "floor, aluminium sliding windows behind steel cages, AC "
              "units, a roll-up style steel door at the entrance.",
    },
    "asia_apt": {
        "a1": "1990s Taiwanese apartment block (gong-yu) clad in beige "
              "rectangular facade tiles, enclosed balconies (loggias) with "
              "aluminium windows and iron grilles, potted plants, laundry "
              "racks, AC units on every floor, a glass-and-aluminium lobby "
              "door with a small canopy.",
        "a2": "1990s Chinese six-storey walk-up residential block: white "
              "render stained grey, aluminium sliding windows with "
              "protruding steel security cages, AC units, a steel "
              "stairwell entrance door.",
        "c1": "Taiwanese street-front building: ground floor of small "
              "shops with roll-up steel shutters (some open with goods "
              "visible, some closed) and blank sign boards, upper floors "
              "clad in grey-white tiles with large aluminium windows.",
        "c2": "Southern Chinese shophouse row: ground floor shop units "
              "with roll-up shutters and glass fronts, blank sign boards, "
              "upper floors of cream render with aluminium sliding windows "
              "behind steel cages and AC units.",
    },
    "ind": {
        "i1": "Modern logistics warehouse: light grey horizontal "
              "trapezoidal steel cladding, a band of horizontal strip "
              "windows high on the wall, loading docks with grey sectional "
              "overhead doors, rubber dock shelters and yellow bollards.",
        "i2": "Industrial hall: blue-grey vertical profiled steel "
              "cladding, few small strip windows, large roller-shutter "
              "doors at the loading bays, grey plinth.",
        "i3": "Factory building of beige precast concrete sandwich panels "
              "with visible joints, larger aluminium windows, sectional "
              "doors at the docks.",
        "i4": "Upper floors of an industrial office wing: off-white "
              "profiled metal cladding with long horizontal ribbon windows "
              "in dark aluminium frames.",
    },
    "africa_res_a": {
        "s1": "West African two-storey family house (Lagos / Accra): "
              "sand-beige painted cement render, steel casement windows "
              "behind decorative flat-bar burglar bars, a painted steel "
              "security door, red dust splashed up the base.",
        "s2": "Nairobi suburban house: terracotta-orange painted render, "
              "steel-framed windows with small panes and flat-bar burglar "
              "bars, a dark green steel door with a small concrete canopy.",
        "s3": "South African township two-storey house: white painted "
              "plaster over block, steel window frames with simple burglar "
              "bars, a steel security gate in front of the door.",
    },
    "africa_res_b": {
        "s1": "Sahel / northern Nigerian town house: ochre-yellow render, "
              "steel windows and doors painted blue-green, decorative "
              "steel grilles, a raised plaster band around each opening.",
        "s2": "Ghanaian house: pale mint-green painted render, glass "
              "louvre (naco) windows behind flat-bar burglar bars, a "
              "painted steel door, faded paint and dust.",
        "s3": "Unplastered house of grey concrete blocks with visible "
              "mortar joints, steel-framed windows with burglar bars, a "
              "steel door, rough cement patches.",
    },
    "africa_apt": {
        "a1": "1970s East African public housing block: cream render with "
              "horizontal painted bands, steel windows with bars, open "
              "breeze-block (claustra) panels at the stairwell, steel "
              "entrance gates.",
        "a2": "Lagos / Accra apartment block: beige render, aluminium "
              "sliding windows behind burglar-proof bars, AC units, painted "
              "steel stairwell doors.",
        "c1": "West African commercial building: ground floor of small "
              "shop units with steel roll-up shutters (some open with "
              "goods) and blank sign boards, upper floors cream render "
              "with steel windows and bars.",
        "c2": "1990s Nairobi commercial building: pale green-grey render, "
              "large tinted-glass aluminium windows, ground-floor shop "
              "fronts behind steel grilles, blank sign boards.",
    },
    "north_america_res_a": {
        "s1": "1990s US suburban house: white vinyl lap siding, white "
              "double-hung windows with faux black shutters, a white "
              "sectional garage door (two bays wide), a six-panel front "
              "door with a small porch light.",
        "s2": "US suburban split-level: pale blue-grey vinyl siding, white "
              "double-hung windows with grids and faux shutters, a "
              "raised-panel garage door, a painted front door.",
        "s3": "US East-Coast brick colonial: red brick in running bond, "
              "white double-hung six-over-six windows with black shutters, "
              "brick soldier lintels, a panelled front door with a "
              "fanlight.",
    },
    "north_america_res_b": {
        "s1": "1970s Canadian bungalow-style house: beige aluminium "
              "siding, double-hung windows with faux shutters, a brown "
              "sectional garage door.",
        "s2": "US ranch house: pale yellow wood clapboard siding with "
              "white trim, double-hung windows, an older wooden garage "
              "door.",
        "s3": "US Southwest stucco house: sand-beige stucco, aluminium "
              "sliding windows, a stucco-framed garage door and a heavy "
              "wooden front door.",
    },
    "north_america_apt": {
        "a1": "1920s US brick walk-up apartment building: dark red brick, "
              "white double-hung windows with stone sills, a glass-panel "
              "entrance door with a small canopy.",
        "a2": "1980s US garden apartment complex: grey-beige vinyl "
              "siding, sliding windows, recessed balconies with wood "
              "railings, metal stair-hall entrance doors.",
        "c1": "US strip-mall / main-street commercial building: beige "
              "stucco (EIFS), large storefront glazing, blank sign bands "
              "above each store, office windows upstairs.",
        "c2": "US main-street brick commercial block: red-brown brick, "
              "cast-iron style storefronts with recessed doors and blank "
              "sign boards, double-hung windows upstairs.",
    },
    "mediterranean_res_a": {
        "s1": "Andalusian / southern Spanish town house: ochre-yellow lime "
              "render, wooden casement windows with green louvred "
              "shutters (some closed), wrought-iron window grilles on the "
              "ground floor, a heavy wooden door.",
        "s2": "Greek island / whitewashed house: white lime render, "
              "blue-painted wooden windows and louvred shutters, a blue "
              "door, cracks and patched whitewash.",
        "s3": "Italian Ligurian house: faded terracotta-pink render with "
              "painted light window surrounds, green louvred persiane "
              "shutters, a wooden door with a stone frame.",
    },
    "mediterranean_res_b": {
        "s1": "Portuguese house: pale sand render, white window frames "
              "with stone surrounds, brown wooden shutters, a small "
              "azulejo-tile panel at the door.",
        "s2": "Southern Italian house: terracotta-red render, brown "
              "louvred shutters, small iron balconettes, a wooden door.",
        "s3": "Spanish 1960s town house: light grey render, aluminium "
              "windows with external persiana roller blinds (some down), "
              "iron grilles on the ground floor.",
    },
    "mediterranean_apt": {
        "a1": "Spanish / Italian 1960s-70s apartment block (palazzina): "
              "cream render, tall French windows with louvred shutters "
              "opening to recessed balconies with iron railings, laundry "
              "and plants, a glass-and-aluminium entrance.",
        "a2": "1980s-90s Greek polykatoikia: white render, deep recessed "
              "balconies with aluminium railings and awnings, sliding "
              "doors with roller shutters, AC units.",
        "c1": "Italian town palazzo with shops: warm ochre render, tall "
              "windows with green shutters, ground-floor shops with "
              "arched openings and blank sign boards.",
        "c2": "1990s Mediterranean commercial building: light grey "
              "render, large aluminium windows, ground-floor glass shop "
              "fronts with blank sign boards.",
    },
    "south_america_res_a": {
        "s1": "Brazilian house: sunflower-yellow painted render, steel "
              "casement windows with white flat-bar grilles, a steel "
              "garage-style front gate, peeling paint.",
        "s2": "Colombian / Peruvian house: cobalt-blue painted render, "
              "casement windows with decorative steel bars, a painted "
              "steel door, cracks and faded paint.",
        "s3": "Argentine house: dusty-pink render, wooden casement windows "
              "with iron bars, wooden persiana shutters, a wooden door.",
    },
    "south_america_res_b": {
        "s1": "Chilean / Argentine house: off-white render, aluminium "
              "windows with iron grilles, a steel security door, grime "
              "splash at the base.",
        "s2": "Latin American auto-constructed house: exposed orange clay "
              "hollow bricks with rough mortar, steel windows with bars, "
              "a steel door, rough cement patches.",
        "s3": "Brazilian house: terracotta-orange painted render, steel "
              "windows with white grilles, a steel door.",
    },
    "south_america_apt": {
        "a1": "Brazilian 1990s apartment block (predio): small white and "
              "beige facade tiles, recessed balconies with glass or steel "
              "railings, aluminium windows, a gated lobby entrance.",
        "a2": "South American 1970s apartment block: cream render, "
              "casement windows with bars, recessed balconies with "
              "laundry, a steel entrance gate.",
        "c1": "South American commercial street building: ochre render, "
              "ground-floor shops with steel roll-up shutters (some open) "
              "and blank sign boards, barred windows upstairs.",
        "c2": "Modern South American commercial building: pale blue-grey "
              "render, large aluminium windows, glass shop fronts with "
              "blank sign boards.",
    },
    "australia_oceania_res_a": {
        "s1": "Australian 1970s cream-brick house: pale cream face brick, "
              "aluminium sliding windows, a flyscreen security door, "
              "brick sills.",
        "s2": "Australian 1960s red/liver-brick house: dark red brick, "
              "aluminium sliding windows, a panelled front door with a "
              "security screen.",
        "s3": "New Zealand / Queensland weatherboard house: white painted "
              "timber weatherboards, white timber casement windows, a "
              "timber front door.",
    },
    "australia_oceania_res_b": {
        "s1": "Australian rendered brick house: off-white bagged render, "
              "aluminium sliding windows, a security screen door.",
        "s2": "Australian fibro cottage: sage-green painted fibro sheet "
              "cladding with cover strips, timber casement windows, a "
              "timber door.",
        "s3": "Queenslander-style house: pale yellow timber "
              "weatherboards, white casement windows with timber hoods, a "
              "timber door.",
    },
    "australia_oceania_apt": {
        "a1": "Australian 1960s-70s walk-up flats (six-pack): cream "
              "brick, large aluminium sliding windows, recessed balconies "
              "with metal balustrades, a glass entry door.",
        "a2": "Australian 2000s apartment block: white render, aluminium "
              "windows, recessed balconies with glass balustrades, a "
              "glass lobby.",
        "c1": "Australian shopping-strip building: beige render, large "
              "shop windows with blank sign boards and a suspended "
              "awning line, aluminium windows upstairs.",
        "c2": "Australian federation-era brick shop row: red-brown "
              "brick, timber shopfronts with blank sign boards, "
              "sash-style windows upstairs.",
    },
    "generic_res_a": {
        "s1": "Plain rendered two-storey house: off-white render, white "
              "casement windows with simple sills, a panelled front door.",
        "s2": "Plain sided house: grey-beige siding, white double-hung "
              "windows, a sectional garage door, a simple front door.",
        "s3": "Plain brick house: red-brown brick, white casement "
              "windows, brick lintels, a painted front door.",
    },
    "generic_res_b": {
        "s1": "Plain rendered house: cream render, white casement "
              "windows, a timber front door.",
        "s2": "Plain concrete-rendered house: light grey render, "
              "aluminium sliding windows, a steel door.",
        "s3": "Plain brick house: light brown brick, white casement "
              "windows, a dark front door.",
    },
    "generic_apt": {
        "a1": "Plain 1970s apartment block: off-white render, casement "
              "windows with roller shutters, recessed balconies with metal "
              "railings, a glass entrance door.",
        "a2": "Plain precast-panel apartment block: grey concrete "
              "panels with joints, large windows, recessed balconies.",
        "c1": "Plain mixed-use building: cream render, ground-floor shop "
              "windows with blank sign boards, casement windows upstairs.",
        "c2": "Plain modern commercial building: light grey render, "
              "large aluminium windows, glass shop fronts with blank sign "
              "boards.",
    },
}
