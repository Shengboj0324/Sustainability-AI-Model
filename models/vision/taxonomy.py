"""
Canonical waste taxonomy + physical-property knowledge base.

This module is the SINGLE SOURCE OF TRUTH for the 30 waste item classes, the
materials they are made of, the disposal bin they belong to, and the physical
properties that make those facts *true*. Everything downstream — the
physics-informed classifier, the evaluation harness, the stress tests, and the
serving code — imports its taxonomy from here so that item / material / bin
predictions can never silently disagree.

Why this exists
---------------
The legacy model used three INDEPENDENT linear heads (item / material / bin)
trained on labels that were themselves derived from a static item->material->bin
lookup. At inference the heads could emit physically impossible combinations
(e.g. material="glass", bin="compost"). Here we encode the lookup explicitly and
expose differentiable mapping matrices so the network can be made consistent
*by construction* (see physics_informed_classifier.py).

Item ordering is FROZEN to match deployment_package/model_metadata.json
(`class_index`, alphabetical 0..29) so existing checkpoints load without
re-indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. ITEM CLASSES  (order frozen to match the trained checkpoint's class_index)
# ---------------------------------------------------------------------------
ITEM_CLASSES: List[str] = [
    "aerosol_cans",                 # 0
    "aluminum_food_cans",           # 1
    "aluminum_soda_cans",           # 2
    "cardboard_boxes",              # 3
    "cardboard_packaging",          # 4
    "clothing",                     # 5
    "coffee_grounds",               # 6
    "disposable_plastic_cutlery",   # 7
    "eggshells",                    # 8
    "food_waste",                   # 9
    "glass_beverage_bottles",       # 10
    "glass_cosmetic_containers",    # 11
    "glass_food_jars",              # 12
    "magazines",                    # 13
    "newspaper",                    # 14
    "office_paper",                 # 15
    "paper_cups",                   # 16
    "plastic_cup_lids",             # 17
    "plastic_detergent_bottles",    # 18
    "plastic_food_containers",      # 19
    "plastic_shopping_bags",        # 20
    "plastic_soda_bottles",         # 21
    "plastic_straws",               # 22
    "plastic_trash_bags",           # 23
    "plastic_water_bottles",        # 24
    "shoes",                        # 25
    "steel_food_cans",              # 26
    "styrofoam_cups",               # 27
    "styrofoam_food_containers",    # 28
    "tea_bags",                     # 29
]
NUM_ITEMS = len(ITEM_CLASSES)
ITEM_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(ITEM_CLASSES)}


# ---------------------------------------------------------------------------
# 2. MATERIAL CLASSES  +  physical properties
# ---------------------------------------------------------------------------
# A material is what gives an item its recyclability physics. We model the
# properties that actually drive disposal decisions and that a vision model can,
# in principle, perceive (transparency, gloss/metallic sheen) or that act as
# priors (density, resin code, magnetism for ferrous metals).

@dataclass(frozen=True)
class MaterialProperty:
    name: str
    resin_code: int          # SPI resin code 1-7 for plastics; 0 = non-plastic
    density_g_cm3: float      # typical bulk density
    transparent: float        # 0..1 visual transparency prior (glass/PET high)
    metallic_sheen: float     # 0..1 metallic appearance prior
    ferromagnetic: float      # 1.0 if steel (magnet test), else 0.0
    curbside_recyclable: float  # 0..1 typically accepted in curbside recycling
    compostable: float          # 0..1
    melting_point_c: float      # informative prior; -1 if N/A (organics/mixed)


# Canonical material set covering all 30 items. Extends the config's 15-material
# list with the physically-real categories the items actually need
# (organic, EPS, textile/leather "mixed") instead of forcing them into "other".
MATERIAL_PROPERTIES: List[MaterialProperty] = [
    #                 name          resin  dens  transp  metal  ferro  recyc  comp  melt
    MaterialProperty("PET",            1, 1.38, 0.85, 0.05, 0.0, 0.95, 0.0, 255.0),
    MaterialProperty("HDPE",           2, 0.95, 0.10, 0.05, 0.0, 0.90, 0.0, 130.0),
    MaterialProperty("PVC",            3, 1.40, 0.20, 0.05, 0.0, 0.20, 0.0, 100.0),
    MaterialProperty("LDPE",           4, 0.92, 0.30, 0.05, 0.0, 0.30, 0.0, 110.0),
    MaterialProperty("PP",             5, 0.91, 0.20, 0.05, 0.0, 0.60, 0.0, 160.0),
    MaterialProperty("PS",             6, 1.05, 0.20, 0.05, 0.0, 0.15, 0.0, 240.0),
    MaterialProperty("EPS",            6, 0.05, 0.05, 0.0,  0.0, 0.05, 0.0, 240.0),  # expanded polystyrene (styrofoam)
    MaterialProperty("glass",          0, 2.50, 0.90, 0.05, 0.0, 0.90, 0.0, 1500.0),
    MaterialProperty("aluminum",       0, 2.70, 0.05, 0.95, 0.0, 0.98, 0.0, 660.0),
    MaterialProperty("steel",          0, 7.85, 0.05, 0.90, 1.0, 0.95, 0.0, 1370.0),
    MaterialProperty("paper",          0, 0.80, 0.05, 0.0,  0.0, 0.85, 0.4, -1.0),
    MaterialProperty("cardboard",      0, 0.70, 0.05, 0.0,  0.0, 0.90, 0.5, -1.0),
    MaterialProperty("coated_paper",   0, 0.85, 0.05, 0.10, 0.0, 0.40, 0.1, -1.0),  # PE-lined cups, glossy magazines
    MaterialProperty("organic",        0, 0.60, 0.05, 0.0,  0.0, 0.0,  0.95, -1.0),
    MaterialProperty("textile",        0, 0.40, 0.05, 0.0,  0.0, 0.10, 0.10, -1.0),  # clothing/cotton/polyester
    MaterialProperty("mixed",          0, 0.80, 0.10, 0.10, 0.0, 0.10, 0.05, -1.0),  # shoes (rubber+textile+metal)
]
MATERIAL_CLASSES: List[str] = [m.name for m in MATERIAL_PROPERTIES]
NUM_MATERIALS = len(MATERIAL_CLASSES)
MATERIAL_TO_IDX: Dict[str, int] = {m: i for i, m in enumerate(MATERIAL_CLASSES)}


# ---------------------------------------------------------------------------
# 3. BIN CLASSES  (disposal destinations — physically real set)
# ---------------------------------------------------------------------------
# The config declared 4 bins (recycle/compost/landfill/hazardous) but the
# authoritative recyclability map needs `special` (store drop-off, e.g. plastic
# film) and `donate` (textiles/shoes). Modeling them correctly is part of the
# physics fix: routing LDPE film to curbside "recycle" is wrong and causes
# real-world contamination.
BIN_CLASSES: List[str] = [
    "recycle",    # 0  curbside recycling
    "compost",    # 1  organics
    "landfill",   # 2  general waste
    "hazardous",  # 3  special hazardous handling
    "special",    # 4  store drop-off / e-waste / film
    "donate",     # 5  reuse / textile recovery
]
NUM_BINS = len(BIN_CLASSES)
BIN_TO_IDX: Dict[str, int] = {b: i for i, b in enumerate(BIN_CLASSES)}


# ---------------------------------------------------------------------------
# 4. ITEM -> (MATERIAL, BIN)   — the ground-truth physics relations
# ---------------------------------------------------------------------------
# Primary material + correct disposal bin for each item. Sourced from
# deployment_package/model_metadata.json `recyclability` and standard
# material-science / municipal-recycling references.

@dataclass(frozen=True)
class ItemFacts:
    material: str
    bin: str
    note: str = ""

ITEM_FACTS: Dict[str, ItemFacts] = {
    "aerosol_cans":               ItemFacts("steel",      "special",  "empty=recycle, pressurized/full=hazardous"),
    "aluminum_food_cans":         ItemFacts("aluminum",   "recycle"),
    "aluminum_soda_cans":         ItemFacts("aluminum",   "recycle"),
    "cardboard_boxes":            ItemFacts("cardboard",  "recycle"),
    "cardboard_packaging":        ItemFacts("cardboard",  "recycle"),
    "clothing":                   ItemFacts("textile",    "donate"),
    "coffee_grounds":             ItemFacts("organic",    "compost"),
    "disposable_plastic_cutlery": ItemFacts("PS",         "landfill"),
    "eggshells":                  ItemFacts("organic",    "compost"),
    "food_waste":                 ItemFacts("organic",    "compost"),
    "glass_beverage_bottles":     ItemFacts("glass",      "recycle"),
    "glass_cosmetic_containers":  ItemFacts("glass",      "recycle"),
    "glass_food_jars":            ItemFacts("glass",      "recycle"),
    "magazines":                  ItemFacts("coated_paper","recycle"),
    "newspaper":                  ItemFacts("paper",      "recycle"),
    "office_paper":               ItemFacts("paper",      "recycle"),
    "paper_cups":                 ItemFacts("coated_paper","landfill", "PE lining blocks recycling in most areas"),
    "plastic_cup_lids":           ItemFacts("PP",         "recycle"),
    "plastic_detergent_bottles":  ItemFacts("HDPE",       "recycle"),
    "plastic_food_containers":    ItemFacts("PET",        "recycle"),
    "plastic_shopping_bags":      ItemFacts("LDPE",       "special",  "film -> store drop-off, not curbside"),
    "plastic_soda_bottles":       ItemFacts("PET",        "recycle"),
    "plastic_straws":             ItemFacts("PP",         "landfill", "too small/light to sort"),
    "plastic_trash_bags":         ItemFacts("LDPE",       "landfill"),
    "plastic_water_bottles":      ItemFacts("PET",        "recycle"),
    "shoes":                      ItemFacts("mixed",      "donate"),
    "steel_food_cans":            ItemFacts("steel",      "recycle"),
    "styrofoam_cups":             ItemFacts("EPS",        "landfill"),
    "styrofoam_food_containers":  ItemFacts("EPS",        "landfill"),
    "tea_bags":                   ItemFacts("organic",    "compost", "some contain PP mesh — check brand"),
}


# ---------------------------------------------------------------------------
# 5. Confusion groups — visually near-identical clusters the model struggles
#    with. Used by the evaluation harness (group accuracy) and the redesign
#    (fine-grained contrastive term targets these).
# ---------------------------------------------------------------------------
CONFUSION_GROUPS: Dict[str, List[str]] = {
    "glass_types":      ["glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars"],
    "cardboard_types":  ["cardboard_boxes", "cardboard_packaging"],
    "aluminum_vs_steel":["aluminum_food_cans", "aluminum_soda_cans", "steel_food_cans", "aerosol_cans"],
    "pet_bottles":      ["plastic_soda_bottles", "plastic_water_bottles"],
    "plastic_film":     ["plastic_shopping_bags", "plastic_trash_bags"],
    "styrofoam_types":  ["styrofoam_cups", "styrofoam_food_containers"],
    "paper_types":      ["newspaper", "office_paper", "magazines", "paper_cups"],
    "organics":         ["coffee_grounds", "eggshells", "food_waste", "tea_bags"],
}


# ---------------------------------------------------------------------------
# 6. Derived index maps  (plain Python; torch tensors built lazily to keep this
#    module importable without a torch dependency for pure-data consumers)
# ---------------------------------------------------------------------------
def item_to_material_index() -> List[int]:
    """idx -> material index, aligned to ITEM_CLASSES order."""
    return [MATERIAL_TO_IDX[ITEM_FACTS[c].material] for c in ITEM_CLASSES]


def item_to_bin_index() -> List[int]:
    """idx -> bin index, aligned to ITEM_CLASSES order."""
    return [BIN_TO_IDX[ITEM_FACTS[c].bin] for c in ITEM_CLASSES]


def material_property_matrix() -> List[List[float]]:
    """(NUM_MATERIALS, n_features) physical-property prior table.

    Feature order: [resin_code/7, density/8, transparent, metallic_sheen,
    ferromagnetic, curbside_recyclable, compostable, melt/1500]. Values are
    normalized to ~[0,1] so they can seed a learnable property-prior embedding.
    """
    rows = []
    for m in MATERIAL_PROPERTIES:
        rows.append([
            m.resin_code / 7.0,
            m.density_g_cm3 / 8.0,
            m.transparent,
            m.metallic_sheen,
            m.ferromagnetic,
            m.curbside_recyclable,
            m.compostable,
            (m.melting_point_c / 1500.0) if m.melting_point_c > 0 else 0.0,
        ])
    return rows


# ----- lazy torch helpers (imported only when torch is available) ----------
def build_mapping_tensors(device=None):
    """Return differentiable-friendly mapping tensors for the consistency layer.

    Returns a dict with:
      item2mat : LongTensor (NUM_ITEMS,)        item idx -> material idx
      item2bin : LongTensor (NUM_ITEMS,)        item idx -> bin idx
      M_item_material : FloatTensor (NUM_ITEMS, NUM_MATERIALS) one-hot rows
      M_item_bin      : FloatTensor (NUM_ITEMS, NUM_BINS)      one-hot rows
      mat_props       : FloatTensor (NUM_MATERIALS, 8) property priors
    The one-hot matrices let us map an item probability distribution to an
    implied material/bin distribution by a single matmul:  p_mat = p_item @ M.
    """
    import torch
    i2m = torch.tensor(item_to_material_index(), dtype=torch.long, device=device)
    i2b = torch.tensor(item_to_bin_index(), dtype=torch.long, device=device)
    M_im = torch.zeros(NUM_ITEMS, NUM_MATERIALS, device=device)
    M_im[torch.arange(NUM_ITEMS), i2m] = 1.0
    M_ib = torch.zeros(NUM_ITEMS, NUM_BINS, device=device)
    M_ib[torch.arange(NUM_ITEMS), i2b] = 1.0
    props = torch.tensor(material_property_matrix(), dtype=torch.float32, device=device)
    return {
        "item2mat": i2m,
        "item2bin": i2b,
        "M_item_material": M_im,
        "M_item_bin": M_ib,
        "mat_props": props,
    }


def disposal_for_item(item_name: str) -> Dict[str, str]:
    """Authoritative, physically-consistent disposal facts for a predicted item.

    The recommended SERVING contract: classify the item, then derive material +
    bin from here rather than trusting weak independent material/bin heads. The
    result is always a legal (item, material, bin) triple. Raises KeyError on an
    unknown item so mismatches fail loud instead of silently mis-routing waste."""
    f = ITEM_FACTS[item_name]
    return {"item": item_name, "material": f.material, "bin": f.bin, "note": f.note}


def validate_taxonomy() -> None:
    """Self-check that every relation is internally consistent. Cheap; call at import time in tests."""
    assert len(ITEM_CLASSES) == NUM_ITEMS == 30, "expected 30 items"
    assert len(set(ITEM_CLASSES)) == NUM_ITEMS, "duplicate item class"
    for c in ITEM_CLASSES:
        assert c in ITEM_FACTS, f"missing facts for {c}"
        f = ITEM_FACTS[c]
        assert f.material in MATERIAL_TO_IDX, f"unknown material {f.material} for {c}"
        assert f.bin in BIN_TO_IDX, f"unknown bin {f.bin} for {c}"
    for g, members in CONFUSION_GROUPS.items():
        for m in members:
            assert m in ITEM_TO_IDX, f"confusion group {g} references unknown item {m}"


if __name__ == "__main__":
    validate_taxonomy()
    print(f"items={NUM_ITEMS} materials={NUM_MATERIALS} bins={NUM_BINS}")
    print("item->material idx:", item_to_material_index())
    print("item->bin idx     :", item_to_bin_index())
    # show the physically-correct material/bin for the worst-failing classes
    for c in ["glass_beverage_bottles", "glass_food_jars", "plastic_trash_bags", "food_waste"]:
        f = ITEM_FACTS[c]
        print(f"  {c:28s} -> material={f.material:12s} bin={f.bin:9s} {f.note}")
