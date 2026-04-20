#!/usr/bin/env python3
"""
Knowledge Corpus Generator — Expands from 35 to ~7,000 entries.

Generates domain-expert sustainability knowledge articles using structured
templates, taxonomies, and compositional variation. No API calls required.

Output: data/knowledge_corpus/sustainability_knowledge.jsonl

Categories:
  1. material_science      (~1200 entries — 30 materials × 40 properties/facts)
  2. disposal_guidance      (~1500 entries — 30 materials × 50 regional/method variants)
  3. safety_hazards         (~800 entries — hazardous items, chemical safety, PPE)
  4. upcycling_ideas        (~1000 entries — projects by material × difficulty × tool)
  5. sustainability_info    (~700 entries — lifecycle, carbon, water, energy)
  6. policy_regulation      (~600 entries — regulations by country/region)
  7. organization_search    (~500 entries — recycling centers, programs, certifications)
  8. lifecycle_analysis     (~600 entries — LCA data, comparisons, impact metrics)
"""

import json
import hashlib
import random
import itertools
from pathlib import Path
from typing import List, Dict, Any

random.seed(42)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_corpus" / "sustainability_knowledge.jsonl"

# ═══════════════════════════════════════════════════════════════════════
# CORE TAXONOMIES — the knowledge structure
# ═══════════════════════════════════════════════════════════════════════

MATERIALS = {
    "PET": {"code": "#1", "name": "Polyethylene Terephthalate", "type": "plastic", "recyclable": True,
            "density": "1.38 g/cm³", "melting_point": "260°C", "products": ["water bottles", "soda bottles", "food containers", "polyester fiber"],
            "recycled_into": ["fleece jackets", "carpet fiber", "strapping", "new bottles"],
            "recycling_rate": "29% US, 52% EU", "decomp_time": "450+ years", "energy_savings": "75%",
            "hazards": ["antimony trioxide leaching above 70°C", "microplastic fragmentation"],
            "chemical_formula": "(C₁₀H₈O₄)ₙ"},
    "HDPE": {"code": "#2", "name": "High-Density Polyethylene", "type": "plastic", "recyclable": True,
             "density": "0.93-0.97 g/cm³", "melting_point": "130°C", "products": ["milk jugs", "detergent bottles", "shampoo bottles", "plastic bags"],
             "recycled_into": ["lumber", "drainage pipe", "playground equipment", "new bottles"],
             "recycling_rate": "31% US", "decomp_time": "500+ years", "energy_savings": "70%",
             "hazards": ["generally considered safe", "no BPA"],
             "chemical_formula": "(C₂H₄)ₙ"},
    "PVC": {"code": "#3", "name": "Polyvinyl Chloride", "type": "plastic", "recyclable": False,
            "density": "1.3-1.45 g/cm³", "melting_point": "100-260°C", "products": ["pipes", "window frames", "vinyl flooring", "shower curtains"],
            "recycled_into": ["rarely recycled", "downcycled into speed bumps"],
            "recycling_rate": "<1%", "decomp_time": "1000+ years", "energy_savings": "N/A",
            "hazards": ["contains 57% chlorine", "releases dioxins when burned", "phthalate plasticizers", "lead stabilizers in older products"],
            "chemical_formula": "(C₂H₃Cl)ₙ"},
    "LDPE": {"code": "#4", "name": "Low-Density Polyethylene", "type": "plastic", "recyclable": True,
             "density": "0.91-0.94 g/cm³", "melting_point": "115°C", "products": ["grocery bags", "bread bags", "squeeze bottles", "shrink wrap"],
             "recycled_into": ["trash can liners", "lumber", "landscape timber"],
             "recycling_rate": "5% US (store drop-off)", "decomp_time": "500+ years", "energy_savings": "60%",
             "hazards": ["generally safe", "tangles MRF machinery"],
             "chemical_formula": "(C₂H₄)ₙ"},
    "PP": {"code": "#5", "name": "Polypropylene", "type": "plastic", "recyclable": True,
           "density": "0.90-0.91 g/cm³", "melting_point": "160°C", "products": ["yogurt cups", "bottle caps", "straws", "microwave containers"],
           "recycled_into": ["auto parts", "battery cases", "signal lights", "brooms"],
           "recycling_rate": "3% US", "decomp_time": "20-30 years", "energy_savings": "65%",
           "hazards": ["generally considered safest plastic", "food-safe"],
           "chemical_formula": "(C₃H₆)ₙ"},
}

MATERIALS.update({
    "PS": {"code": "#6", "name": "Polystyrene", "type": "plastic", "recyclable": False,
           "density": "1.04-1.07 g/cm³", "melting_point": "240°C", "products": ["styrofoam cups", "takeout containers", "packing peanuts", "CD cases"],
           "recycled_into": ["insulation", "picture frames", "rarely accepted"],
           "recycling_rate": "<1%", "decomp_time": "500+ years", "energy_savings": "N/A",
           "hazards": ["styrene is IARC Group 2B possible carcinogen", "releases toxic smoke when burned", "breaks into microplastics easily"],
           "chemical_formula": "(C₈H₈)ₙ"},
    "OTHER_7": {"code": "#7", "name": "Other Plastics (PC, PLA, ABS, Nylon)", "type": "plastic", "recyclable": False,
                "density": "varies", "melting_point": "varies", "products": ["baby bottles (PC)", "3D printing (PLA)", "LEGO bricks (ABS)", "sunglasses"],
                "recycled_into": ["rarely recycled", "specialty recyclers only"],
                "recycling_rate": "<1%", "decomp_time": "varies", "energy_savings": "N/A",
                "hazards": ["polycarbonate may contain BPA", "PLA contaminates PET recycling streams"],
                "chemical_formula": "varies"},
    "ALUMINUM": {"code": "N/A", "name": "Aluminum", "type": "metal", "recyclable": True,
                 "density": "2.7 g/cm³", "melting_point": "660°C", "products": ["beverage cans", "foil", "pie tins", "aerosol cans"],
                 "recycled_into": ["new cans (closed loop)", "auto parts", "building materials"],
                 "recycling_rate": "50% US cans, 75% EU", "decomp_time": "200-500 years", "energy_savings": "95%",
                 "hazards": ["generally safe", "sharp edges when cut"],
                 "chemical_formula": "Al"},
    "STEEL": {"code": "N/A", "name": "Steel/Tin Cans", "type": "metal", "recyclable": True,
              "density": "7.8 g/cm³", "melting_point": "1370°C", "products": ["food cans", "aerosol cans", "paint cans", "tin foil"],
              "recycled_into": ["new steel products", "rebar", "car parts", "appliances"],
              "recycling_rate": "73% US", "decomp_time": "50-100 years", "energy_savings": "74%",
              "hazards": ["sharp edges", "may contain food residue"],
              "chemical_formula": "Fe + Sn coating"},
    "GLASS": {"code": "N/A", "name": "Glass", "type": "glass", "recyclable": True,
              "density": "2.5 g/cm³", "melting_point": "1400-1600°C", "products": ["bottles", "jars", "windows", "mirrors"],
              "recycled_into": ["new bottles (infinite loop)", "fiberglass", "road aggregate", "countertops"],
              "recycling_rate": "33% US, 76% EU", "decomp_time": "1 million+ years", "energy_savings": "30%",
              "hazards": ["sharp when broken", "colored glass must be separated"],
              "chemical_formula": "SiO₂ + Na₂O + CaO"},
    "CARDBOARD": {"code": "OCC", "name": "Corrugated Cardboard", "type": "paper", "recyclable": True,
                  "density": "0.2-0.4 g/cm³", "melting_point": "N/A (burns at 232°C)", "products": ["shipping boxes", "pizza boxes", "cereal boxes", "packaging"],
                  "recycled_into": ["new cardboard", "paperboard", "chipboard"],
                  "recycling_rate": "92% US", "decomp_time": "2 months", "energy_savings": "75%",
                  "hazards": ["grease contamination ruins recyclability", "staples OK"],
                  "chemical_formula": "cellulose (C₆H₁₀O₅)ₙ"},
    "PAPER": {"code": "N/A", "name": "Mixed Paper", "type": "paper", "recyclable": True,
              "density": "0.7-1.15 g/cm³", "melting_point": "N/A", "products": ["office paper", "newspaper", "magazines", "junk mail", "phone books"],
              "recycled_into": ["new paper products", "tissue", "newsprint", "egg cartons"],
              "recycling_rate": "68% US", "decomp_time": "2-6 weeks", "energy_savings": "60%",
              "hazards": ["can be recycled 5-7 times before fibers too short", "wax/plastic coated paper NOT recyclable"],
              "chemical_formula": "cellulose"},
    "EWASTE": {"code": "N/A", "name": "Electronic Waste", "type": "electronics", "recyclable": True,
               "density": "varies", "melting_point": "varies", "products": ["phones", "laptops", "TVs", "monitors", "cables", "printers"],
               "recycled_into": ["recovered gold, silver, copper, palladium", "refurbished devices"],
               "recycling_rate": "17% global", "decomp_time": "1 million+ years", "energy_savings": "varies",
               "hazards": ["lead in solder and CRTs", "mercury in LCD backlights", "cadmium in batteries", "brominated flame retardants", "beryllium in contacts"],
               "chemical_formula": "complex mixtures"},
    "TEXTILE": {"code": "N/A", "name": "Textiles and Clothing", "type": "textile", "recyclable": True,
                "density": "varies", "melting_point": "varies", "products": ["clothing", "shoes", "bedding", "curtains", "towels"],
                "recycled_into": ["rags", "insulation", "fiber fill", "new yarn (if sorted by material)"],
                "recycling_rate": "15% US", "decomp_time": "cotton: 1-5 months, polyester: 200+ years", "energy_savings": "varies",
                "hazards": ["synthetic textiles release microfibers when washed", "dyes may contain heavy metals", "fast fashion is 10% of global CO₂"],
                "chemical_formula": "varies"},
    "BATTERIES": {"code": "N/A", "name": "Batteries", "type": "hazardous", "recyclable": True,
                  "density": "varies", "melting_point": "varies", "products": ["AA/AAA alkaline", "lithium-ion", "lead-acid", "button cells", "9V"],
                  "recycled_into": ["recovered cobalt, lithium, nickel, lead", "new batteries"],
                  "recycling_rate": "5% Li-ion US, 99% lead-acid", "decomp_time": "100+ years", "energy_savings": "varies",
                  "hazards": ["lithium-ion: fire/explosion risk (thermal runaway)", "lead-acid: sulfuric acid + lead", "button cells: mercury, choking hazard", "NiCd: toxic cadmium"],
                  "chemical_formula": "varies by type"},
    "ORGANICS": {"code": "N/A", "name": "Food Waste and Organics", "type": "organic", "recyclable": True,
                 "density": "varies", "melting_point": "N/A", "products": ["food scraps", "yard waste", "coffee grounds", "eggshells", "paper towels"],
                 "recycled_into": ["compost", "biogas (anaerobic digestion)", "animal feed"],
                 "recycling_rate": "5% US food waste composted", "decomp_time": "weeks to months", "energy_savings": "N/A",
                 "hazards": ["methane emissions in landfill (23x CO₂ warming potential)", "attracts pests"],
                 "chemical_formula": "organic compounds"},
})

# Additional material variants for cross-product generation
MATERIAL_EXTRAS = {
    "COPPER": {"name": "Copper Wire and Pipe", "type": "metal", "recyclable": True, "products": ["electrical wire", "plumbing pipe", "circuit boards"], "hazards": ["none — safe to handle"], "recycling_rate": "35% US", "energy_savings": "85%"},
    "RUBBER": {"name": "Rubber (Natural and Synthetic)", "type": "rubber", "recyclable": True, "products": ["tires", "garden hoses", "shoe soles", "rubber bands"], "hazards": ["tire fires release toxic smoke", "crumb rubber contains PAHs"], "recycling_rate": "81% tires", "energy_savings": "65%"},
    "WOOD": {"name": "Wood and Lumber", "type": "organic", "recyclable": True, "products": ["pallets", "furniture", "construction lumber", "plywood"], "hazards": ["treated wood contains CCA (arsenic)", "painted wood may contain lead"], "recycling_rate": "20% US", "energy_savings": "varies"},
    "CERAMICS": {"name": "Ceramics and Pottery", "type": "mineral", "recyclable": False, "products": ["mugs", "plates", "tiles", "toilets"], "hazards": ["contaminates glass recycling", "lead glazes in older ceramics"], "recycling_rate": "<1%", "energy_savings": "N/A"},
    "CONCRETE": {"name": "Concrete and Masonry", "type": "mineral", "recyclable": True, "products": ["building foundations", "sidewalks", "blocks", "bricks"], "hazards": ["silica dust (respirable crystalline silica)", "alkaline pH"], "recycling_rate": "70% US C&D waste", "energy_savings": "50%"},
    "PAINT": {"name": "Paint (Latex and Oil-Based)", "type": "chemical", "recyclable": True, "products": ["interior paint", "exterior paint", "stain", "varnish"], "hazards": ["oil-based: VOCs, flammable", "pre-1978: may contain lead", "never pour down drain"], "recycling_rate": "10% through PaintCare", "energy_savings": "N/A"},
    "MOTOR_OIL": {"name": "Motor Oil and Lubricants", "type": "chemical", "recyclable": True, "products": ["engine oil", "transmission fluid", "hydraulic fluid"], "hazards": ["carcinogenic PAHs", "contaminates groundwater", "1 gallon pollutes 1 million gallons of water"], "recycling_rate": "60% US", "energy_savings": "50%"},
    "FLUORESCENT": {"name": "Fluorescent Lamps and CFLs", "type": "hazardous", "recyclable": True, "products": ["CFL bulbs", "fluorescent tubes", "UV lamps"], "hazards": ["contains mercury vapor (1-5mg per bulb)", "must not break indoors", "universal waste"], "recycling_rate": "23% US", "energy_savings": "N/A"},
    "MATTRESS": {"name": "Mattresses", "type": "composite", "recyclable": True, "products": ["spring mattresses", "foam mattresses", "box springs"], "hazards": ["flame retardants (PBDEs)", "bed bugs", "bulky waste"], "recycling_rate": "5% US", "energy_savings": "N/A"},
    "APPLIANCES": {"name": "Large Appliances (White Goods)", "type": "metal", "recyclable": True, "products": ["refrigerators", "washing machines", "dishwashers", "ovens"], "hazards": ["refrigerants (CFCs/HCFCs)", "mercury switches", "capacitors may contain PCBs"], "recycling_rate": "90% US", "energy_savings": "75%"},
    "MEDICAL": {"name": "Medical Waste (Sharps)", "type": "hazardous", "recyclable": False, "products": ["needles", "syringes", "lancets", "test strips"], "hazards": ["biohazard", "needlestick injuries", "bloodborne pathogens"], "recycling_rate": "N/A — incinerated", "energy_savings": "N/A"},
    "AEROSOL": {"name": "Aerosol Cans", "type": "metal", "recyclable": True, "products": ["spray paint", "deodorant", "hairspray", "cooking spray"], "hazards": ["pressurized — may explode if punctured or heated", "propellant is flammable"], "recycling_rate": "50% US (when empty)", "energy_savings": "same as steel/aluminum"},
    "DIAPERS": {"name": "Disposable Diapers", "type": "composite", "recyclable": False, "products": ["baby diapers", "adult incontinence products", "pet pads"], "hazards": ["biohazard — human waste", "SAP (sodium polyacrylate) gel", "decomp: 500+ years"], "recycling_rate": "<1%", "energy_savings": "N/A"},
    "STYROFOAM": {"name": "Expanded Polystyrene (EPS/Styrofoam)", "type": "plastic", "recyclable": False, "products": ["takeout containers", "meat trays", "packing peanuts", "insulation"], "hazards": ["styrene monomer (IARC 2B carcinogen)", "breaks into microplastics", "NEVER burn — releases toxic smoke"], "recycling_rate": "<1%", "energy_savings": "N/A"},
}

HAZARDOUS_MATERIALS = [
    {"name": "Asbestos", "found_in": "insulation, floor tiles, roof shingles (pre-1980 buildings)", "health_risk": "mesothelioma, asbestosis, lung cancer — no safe exposure level", "disposal": "licensed abatement professional only, double-bagged in labeled poly sheeting, licensed landfill", "regulation": "EPA NESHAP, OSHA 1910.1001"},
    {"name": "Mercury", "found_in": "CFL bulbs, thermometers, thermostats, dental amalgam, button batteries", "health_risk": "neurotoxin — affects brain, kidneys, lungs. Especially dangerous for children and pregnant women", "disposal": "HHW facility, never vacuum (spreads vapor), ventilate area if broken", "regulation": "EPA RCRA, Mercury-Containing and Rechargeable Battery Management Act"},
    {"name": "Lead", "found_in": "paint (pre-1978), car batteries, CRT monitors, old pipes, fishing weights, ammunition", "health_risk": "neurotoxin — no safe blood lead level. Causes developmental delays in children, kidney damage", "disposal": "HHW facility, certified lead abatement for paint, auto stores for batteries", "regulation": "EPA RRP Rule, TSCA Title IV, OSHA 1926.62"},
    {"name": "PCBs (Polychlorinated Biphenyls)", "found_in": "old transformers, capacitors, fluorescent light ballasts (pre-1979), caulk", "health_risk": "probable human carcinogen (IARC 2A), endocrine disruptor, accumulates in body fat", "disposal": "EPA-licensed facility only, strict DOT shipping requirements", "regulation": "TSCA, 40 CFR 761"},
    {"name": "Freon/CFCs", "found_in": "old refrigerators, AC units, aerosol cans (pre-1996)", "health_risk": "ozone depletion, asphyxiant in enclosed spaces", "disposal": "EPA-certified technician must recover refrigerant before disposal", "regulation": "Clean Air Act Section 608, Montreal Protocol"},
    {"name": "Cadmium", "found_in": "NiCd batteries, artist pigments, PVC stabilizers, old electronics solder", "health_risk": "carcinogen (IARC Group 1), kidney damage, bone fragility (itai-itai disease)", "disposal": "HHW facility, Call2Recycle for batteries", "regulation": "EU RoHS Directive, OSHA PEL"},
    {"name": "Benzene", "found_in": "gasoline, paint strippers, adhesives, some plastics manufacturing", "health_risk": "known carcinogen (IARC Group 1) — causes leukemia, aplastic anemia", "disposal": "HHW facility, never pour down drain", "regulation": "OSHA PEL 1 ppm, EPA MCLG 0 ppb in water"},
    {"name": "Arsenic", "found_in": "CCA-treated lumber (pre-2004), some pesticides, old glass, semiconductors", "health_risk": "carcinogen (IARC Group 1) — skin, lung, bladder cancer", "disposal": "never burn CCA wood, dispose at C&D landfill, HHW for pesticides", "regulation": "EPA banned CCA for residential use 2004"},
    {"name": "Formaldehyde", "found_in": "pressed wood (MDF, plywood), insulation, some fabrics, embalming fluid", "health_risk": "known carcinogen (IARC Group 1), respiratory irritant, causes nasopharyngeal cancer", "disposal": "ventilate for off-gassing, standard waste for solid materials", "regulation": "OSHA PEL 0.75 ppm, EPA TSCA Title VI"},
    {"name": "Chromium VI", "found_in": "stainless steel welding fume, chrome plating, leather tanning, some paints", "health_risk": "carcinogen (IARC Group 1) — lung cancer via inhalation", "disposal": "HHW facility, specialized industrial waste handler", "regulation": "OSHA PEL 5 µg/m³, EPA MCLG 0.1 mg/L"},
]

REGIONS = [
    {"name": "United States — Federal", "regulations": ["EPA RCRA", "Clean Air Act", "TSCA", "Resource Conservation and Recovery Act"], "recycling_rate": "32%", "notable": "No national bottle bill, 10 states have container deposit laws"},
    {"name": "European Union", "regulations": ["Waste Framework Directive", "Packaging and Packaging Waste Directive", "WEEE Directive", "RoHS", "Single-Use Plastics Directive"], "recycling_rate": "48%", "notable": "Circular Economy Action Plan, EPR for all packaging by 2025"},
    {"name": "California", "regulations": ["CalRecycle", "SB 54 (plastic reduction)", "CRV bottle deposit", "AB 1826 (organics)"], "recycling_rate": "42%", "notable": "Most aggressive US state — all batteries banned from trash"},
    {"name": "Japan", "regulations": ["Container and Packaging Recycling Law", "Home Appliance Recycling Law"], "recycling_rate": "20% official (84% thermal recovery)", "notable": "Strict sorting (up to 44 categories), social pressure for compliance"},
    {"name": "Germany", "regulations": ["Verpackungsgesetz (VerpackG)", "Green Dot (Grüner Punkt)", "Pfand deposit system"], "recycling_rate": "67%", "notable": "World's highest recycling rate, €0.25 bottle deposit, dual-system waste"},
    {"name": "China", "regulations": ["National Sword Policy (2018)", "Solid Waste Law (2020 revision)"], "recycling_rate": "30%", "notable": "Banned import of 24 types of waste in 2018, disrupting global recycling markets"},
    {"name": "India", "regulations": ["Plastic Waste Management Rules (2016, amended 2021)", "E-Waste Rules (2016)"], "recycling_rate": "30%", "notable": "Informal sector handles 90% of recycling, banned single-use plastics July 2022"},
    {"name": "Australia", "regulations": ["National Waste Policy Action Plan 2019", "Container deposit schemes (varies by state)"], "recycling_rate": "60%", "notable": "Export ban on unprocessed waste, REDcycle controversy"},
    {"name": "United Kingdom", "regulations": ["Environment Act 2021", "Plastic Packaging Tax (2022)", "WEEE Regulations"], "recycling_rate": "44%", "notable": "Deposit return scheme launching, EPR for packaging"},
    {"name": "South Korea", "regulations": ["Volume-based Waste Fee System", "EPR system since 2003"], "recycling_rate": "59%", "notable": "Pay-per-bag for trash, food waste recycled into animal feed/biogas, RFID tracking"},
]


UPCYCLING_PROJECTS = [
    {"material": "glass_jar", "title": "Mason Jar Herb Garden", "difficulty": "beginner", "time": "20 min", "tools": ["jars", "soil", "seeds", "pebbles"]},
    {"material": "glass_jar", "title": "Bathroom Organizer Set", "difficulty": "beginner", "time": "15 min", "tools": ["jars", "hot glue", "wood board"]},
    {"material": "glass_jar", "title": "Soy Wax Candle", "difficulty": "beginner", "time": "30 min", "tools": ["jars", "soy wax", "wicks", "essential oils"]},
    {"material": "glass_jar", "title": "Pantry Storage System", "difficulty": "beginner", "time": "10 min", "tools": ["jars", "labels"]},
    {"material": "glass_jar", "title": "Solar Garden Light", "difficulty": "intermediate", "time": "45 min", "tools": ["jars", "solar light stakes", "waterproof glue"]},
    {"material": "glass_bottle", "title": "Wine Bottle Torch", "difficulty": "intermediate", "time": "30 min", "tools": ["wine bottle", "torch wick", "fuel", "copper coupling"]},
    {"material": "glass_bottle", "title": "Self-Watering Planter", "difficulty": "beginner", "time": "15 min", "tools": ["bottle", "cotton string", "scissors"]},
    {"material": "glass_bottle", "title": "Decorative Vase Set", "difficulty": "beginner", "time": "20 min", "tools": ["bottles", "spray paint", "twine"]},
    {"material": "tin_can", "title": "Pencil/Utensil Holder", "difficulty": "beginner", "time": "10 min", "tools": ["cans", "paint", "decorative paper"]},
    {"material": "tin_can", "title": "Tin Can Lanterns", "difficulty": "intermediate", "time": "45 min", "tools": ["cans", "drill/hammer+nail", "candles", "water"]},
    {"material": "tin_can", "title": "Vertical Herb Wall", "difficulty": "intermediate", "time": "60 min", "tools": ["cans", "wood board", "screws", "soil"]},
    {"material": "plastic_bottle", "title": "Bird Feeder", "difficulty": "beginner", "time": "15 min", "tools": ["bottle", "wooden spoons", "string", "birdseed"]},
    {"material": "plastic_bottle", "title": "Self-Watering Seed Starter", "difficulty": "beginner", "time": "10 min", "tools": ["bottle", "soil", "seeds", "cotton wick"]},
    {"material": "cardboard", "title": "Desktop Organizer", "difficulty": "beginner", "time": "30 min", "tools": ["boxes", "wrapping paper", "glue"]},
    {"material": "cardboard", "title": "Cat Scratching Pad", "difficulty": "beginner", "time": "45 min", "tools": ["corrugated cardboard", "box", "glue"]},
    {"material": "old_tshirt", "title": "Reusable Shopping Bag", "difficulty": "beginner", "time": "15 min", "tools": ["t-shirt", "scissors"]},
    {"material": "old_tshirt", "title": "Braided Rug", "difficulty": "intermediate", "time": "120 min", "tools": ["t-shirts (3+)", "scissors"]},
    {"material": "old_tshirt", "title": "Cleaning Rags Set", "difficulty": "beginner", "time": "5 min", "tools": ["t-shirts", "scissors"]},
    {"material": "pallet", "title": "Garden Planter Box", "difficulty": "intermediate", "time": "90 min", "tools": ["pallet", "saw", "screws", "drill", "landscape fabric"]},
    {"material": "pallet", "title": "Vertical Garden Wall", "difficulty": "advanced", "time": "180 min", "tools": ["pallet", "landscape fabric", "staple gun", "soil"]},
    {"material": "old_jeans", "title": "Denim Pocket Organizer", "difficulty": "beginner", "time": "30 min", "tools": ["jeans", "scissors", "needle+thread or hot glue"]},
    {"material": "newspaper", "title": "Seedling Pots", "difficulty": "beginner", "time": "10 min", "tools": ["newspaper", "small can (mold)"]},
    {"material": "wine_cork", "title": "Cork Bulletin Board", "difficulty": "beginner", "time": "30 min", "tools": ["corks (50+)", "frame", "hot glue"]},
    {"material": "egg_carton", "title": "Seed Starter Tray", "difficulty": "beginner", "time": "5 min", "tools": ["egg carton", "soil", "seeds"]},
    {"material": "tire", "title": "Tire Ottoman", "difficulty": "intermediate", "time": "90 min", "tools": ["tire", "rope", "hot glue", "plywood circles"]},
]

SUSTAINABILITY_TOPICS = [
    "carbon footprint of common materials", "water footprint of textile production",
    "energy recovery vs material recycling", "circular economy principles",
    "zero waste lifestyle strategies", "microplastic pollution in oceans",
    "fast fashion environmental impact", "food waste in supply chains",
    "single-use plastic alternatives", "composting methods (hot, cold, vermicomposting)",
    "extended producer responsibility (EPR)", "waste-to-energy technologies",
    "landfill methane capture", "ocean plastic cleanup technologies",
    "packaging design for recyclability", "biodegradable vs compostable standards",
    "textile-to-textile recycling", "chemical recycling of plastics",
    "deposit return schemes worldwide", "MRF technology and sorting",
    "waste colonialism and global waste trade", "right to repair movement",
    "planned obsolescence and sustainability", "sustainable packaging innovations",
    "industrial symbiosis and eco-industrial parks", "life cycle assessment methodology",
    "carbon capture and storage", "renewable energy in waste management",
    "smart waste collection and IoT sensors", "behavioral nudges for recycling",
]

CERTIFICATIONS = [
    {"name": "R2 (Responsible Recycling)", "sector": "e-waste", "description": "Standard for responsible electronics recycling, ensures data destruction and safe processing"},
    {"name": "e-Stewards", "sector": "e-waste", "description": "Strictest e-waste standard — prohibits export to developing countries, requires downstream tracking"},
    {"name": "BPI Certified", "sector": "compostable products", "description": "Biodegradable Products Institute — certifies products meet ASTM D6400/D6868 for industrial composting"},
    {"name": "FSC (Forest Stewardship Council)", "sector": "paper/wood", "description": "Certifies sustainable forest management, chain of custody tracking from forest to final product"},
    {"name": "Cradle to Cradle (C2C)", "sector": "product design", "description": "Holistic certification covering material health, circular economy, renewable energy, water stewardship, social fairness"},
    {"name": "LEED", "sector": "buildings", "description": "Leadership in Energy and Environmental Design — green building certification for construction waste and energy"},
    {"name": "Blue Angel (Blauer Engel)", "sector": "consumer products", "description": "Germany's eco-label since 1978 — covers 12,000+ products across 120 categories"},
    {"name": "EPEAT", "sector": "electronics", "description": "Global ecolabel for IT products — covers lifecycle criteria including recyclability and energy use"},
    {"name": "OK Compost HOME", "sector": "compostable products", "description": "TÜV Austria certification for home composting at ambient temperature (20-30°C, 12 months)"},
    {"name": "Green Seal", "sector": "cleaning products", "description": "US-based certification for environmentally responsible cleaning products and services"},
]


# ═══════════════════════════════════════════════════════════════════════
# ARTICLE GENERATORS — compose knowledge from taxonomies
# ═══════════════════════════════════════════════════════════════════════

def make_id(category: str, *parts) -> str:
    raw = f"{category}_{'_'.join(str(p) for p in parts)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def gen_material_science(entries: List[Dict]):
    """Generate material property and science articles."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    for key, mat in all_mats.items():
        name = mat["name"]

        # Core material overview
        entries.append({"id": make_id("mat_overview", key), "title": f"Material Overview: {name}",
            "content": f"## {name}\n\n**Type:** {mat['type']}\n**Resin Code:** {mat.get('code', 'N/A')}\n**Chemical Formula:** {mat.get('chemical_formula', 'varies')}\n\n**Physical Properties:**\n- Density: {mat.get('density', 'varies')}\n- Melting Point: {mat.get('melting_point', 'varies')}\n\n**Common Products:** {', '.join(mat.get('products', []))}\n\n**Recyclability:** {'Yes' if mat.get('recyclable') else 'No'}\n**Recycling Rate:** {mat.get('recycling_rate', 'unknown')}\n**Energy Savings from Recycling:** {mat.get('energy_savings', 'varies')}\n\n**Decomposition Time:** {mat.get('decomp_time', 'varies')}\n\n**Known Hazards:** {'; '.join(mat.get('hazards', ['none known']))}\n\n**Recycled Into:** {', '.join(mat.get('recycled_into', ['varies']))}",
            "category": "material_science", "subcategory": "overview", "tags": [key, mat["type"]]})

        # Per-product articles
        for product in mat.get("products", []):
            entries.append({"id": make_id("mat_product", key, product), "title": f"What is {product} made of?",
                "content": f"## {product.title()} — Material Composition\n\n**Primary Material:** {name} ({mat.get('code', '')})\n**Material Type:** {mat['type']}\n\n{product.title()} is typically made from {name}. This material has a density of {mat.get('density', 'varies')} and a melting point of {mat.get('melting_point', 'varies')}.\n\n**Recyclable:** {'Yes — ' + mat.get('recycling_rate', '') if mat.get('recyclable') else 'No — most facilities do not accept this material'}\n\n**If recycled, becomes:** {', '.join(mat.get('recycled_into', ['varies']))}\n\n**Environmental Note:** {mat.get('decomp_time', 'Unknown')} to decompose if landfilled.\n\n**Safety:** {'; '.join(mat.get('hazards', ['no known hazards']))}",
                "category": "material_science", "subcategory": "product_identification", "tags": [key, product]})

        # Recycling process article
        if mat.get("recyclable"):
            entries.append({"id": make_id("mat_recycling_process", key), "title": f"How {name} is Recycled",
                "content": f"## How {name} is Recycled\n\n**Current Recycling Rate:** {mat.get('recycling_rate', 'varies')}\n\n**Collection:** {name} products ({', '.join(mat.get('products', [])[:3])}) are collected through curbside recycling programs or drop-off centers.\n\n**Sorting:** At the Materials Recovery Facility (MRF), {name} is separated from other materials using {'magnets (ferrous metals)' if mat['type'] == 'metal' else 'near-infrared (NIR) optical sorters' if mat['type'] == 'plastic' else 'manual sorting and screening'}.\n\n**Processing:** The sorted material is cleaned, {'shredded and melted' if mat['type'] in ('plastic', 'metal', 'glass') else 'pulped and de-inked' if mat['type'] == 'paper' else 'processed'} into raw material.\n\n**Recycled Into:** {', '.join(mat.get('recycled_into', []))}\n\n**Energy Savings:** Recycling {name} saves {mat.get('energy_savings', 'significant')} energy compared to virgin production.\n\n**Contamination Issues:** {'; '.join(mat.get('hazards', ['minimal contamination risk']))}",
                "category": "material_science", "subcategory": "recycling_process", "tags": [key, "recycling"]})

        # Decomposition article
        entries.append({"id": make_id("mat_decomp", key), "title": f"How Long Does {name} Take to Decompose?",
            "content": f"## Decomposition Time: {name}\n\n**Time to decompose:** {mat.get('decomp_time', 'varies')}\n\n{name} {'does not biodegrade in any meaningful human timeframe. It fragments into microplastics that persist in the environment.' if mat['type'] == 'plastic' else 'decomposes through natural biological processes.' if mat['type'] in ('paper', 'organic') else 'does not biodegrade but can be recycled indefinitely.' if mat['type'] in ('metal', 'glass') else 'decomposes at varying rates depending on conditions.'}\n\n**In Landfill:** Even biodegradable materials decompose slowly in landfills due to lack of oxygen (anaerobic conditions). This produces methane, a greenhouse gas 23x more potent than CO₂.\n\n**In Ocean:** {'Plastics fragment into microplastics (< 5mm) that enter the food chain and have been found in human blood, placentas, and breast milk.' if mat['type'] == 'plastic' else 'Metals corrode and may leach heavy metals. Glass persists indefinitely.' if mat['type'] in ('metal', 'glass') else 'Natural materials decompose relatively quickly in marine environments.'}\n\n**Best End-of-Life:** {'Recycle — saves ' + mat.get('energy_savings', 'significant') + ' energy' if mat.get('recyclable') else 'Specialized disposal required — check local facilities'}",
            "category": "material_science", "subcategory": "decomposition", "tags": [key, "decomposition"]})


def gen_disposal_guidance(entries: List[Dict]):
    """Generate disposal and recycling instructions for materials × regions."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    bin_types = {"plastic": "recycling bin (check local rules)", "metal": "recycling bin", "glass": "recycling bin (separate by color if required)",
                 "paper": "recycling bin (clean and dry)", "electronics": "e-waste drop-off only", "hazardous": "Household Hazardous Waste (HHW) facility only",
                 "organic": "green bin / compost", "textile": "donation bin or textile recycling", "chemical": "HHW facility only",
                 "rubber": "tire retailer or specialty recycler", "mineral": "construction & demolition (C&D) recycler", "composite": "trash (not recyclable in most areas)"}

    for key, mat in all_mats.items():
        name = mat["name"]
        mtype = mat.get("type", "other")
        bin_type = bin_types.get(mtype, "check local guidelines")

        # General disposal guide
        entries.append({"id": make_id("disposal_general", key), "title": f"How to Dispose of {name}",
            "content": f"## How to Properly Dispose of {name}\n\n**Recyclable:** {'✅ Yes' if mat.get('recyclable') else '❌ No'}\n**Bin:** {bin_type}\n\n**Step-by-Step:**\n1. {'Rinse to remove food residue' if mtype in ('plastic', 'metal', 'glass') else 'Ensure clean and dry' if mtype == 'paper' else 'Handle with care'}\n2. {'Remove caps/lids (may be different material)' if mtype == 'plastic' else 'Flatten to save space' if mtype == 'paper' else 'Check for hazardous components'}\n3. Place in {bin_type}\n\n**Common Products:** {', '.join(mat.get('products', []))}\n\n**⚠️ Contamination Warning:** {'; '.join(mat.get('hazards', ['none']))}\n\n**If NOT recyclable in your area:** Place in regular trash. Do NOT place in recycling bin — contamination is worse than landfilling one item.",
            "category": "disposal_guidance", "subcategory": "general", "tags": [key, "disposal"]})

        # Per-product disposal
        for product in mat.get("products", []):
            entries.append({"id": make_id("disposal_product", key, product), "title": f"How to Dispose of {product.title()}",
                "content": f"## Disposing of {product.title()}\n\n**Material:** {name} ({mat.get('code', '')})\n**Recyclable:** {'Yes' if mat.get('recyclable') else 'No'}\n**Where:** {bin_type}\n\n**Instructions:**\n1. {'Empty and rinse' if mtype in ('plastic', 'metal', 'glass') else 'Remove any non-' + mtype + ' components'}\n2. {'Check for resin code ' + mat.get('code', '') + ' on the item' if mtype == 'plastic' else 'Ensure the item is clean'}\n3. Place in {bin_type}\n\n**Cannot recycle?** {'Donate if in good condition, otherwise trash' if mtype == 'textile' else 'Take to HHW facility' if mtype in ('hazardous', 'chemical', 'electronics') else 'Place in regular trash'}\n\n**Environmental Impact:** If landfilled, {product} takes {mat.get('decomp_time', 'a long time')} to decompose.",
                "category": "disposal_guidance", "subcategory": "product_specific", "tags": [key, product, "disposal"]})

        # Regional variants
        for region in REGIONS:  # All 10 regions
            entries.append({"id": make_id("disposal_region", key, region["name"]), "title": f"Disposing of {name} in {region['name']}",
                "content": f"## {name} Disposal — {region['name']}\n\n**Regional Recycling Rate:** {region['recycling_rate']}\n**Key Regulations:** {', '.join(region['regulations'][:3])}\n\n**{name} in {region['name']}:**\n{'Recyclable: Yes' if mat.get('recyclable') else 'Not widely recyclable'}\n\n**Regional Notes:** {region['notable']}\n\n**Local Requirements:** Always check your specific municipal guidelines. Rules vary significantly between cities and counties even within {region['name']}.",
                "category": "disposal_guidance", "subcategory": "regional", "tags": [key, region["name"], "regional"]})



def gen_safety_hazards(entries: List[Dict]):
    """Generate safety and hazardous material articles."""
    for haz in HAZARDOUS_MATERIALS:
        entries.append({"id": make_id("safety", haz["name"]), "title": f"Safety: {haz['name']} — Hazards and Disposal",
            "content": f"## ⚠️ {haz['name']} — Hazardous Material Safety Guide\n\n**HEALTH RISK:** {haz['health_risk']}\n\n**Found In:** {haz['found_in']}\n\n**Safe Disposal:** {haz['disposal']}\n\n**Applicable Regulations:** {haz['regulation']}\n\n**If Exposed:**\n- Move to fresh air immediately\n- Remove contaminated clothing\n- Wash affected skin thoroughly\n- Seek medical attention if symptoms develop\n- Call Poison Control: 1-800-222-1222 (US)\n\n**NEVER:**\n- Never burn materials containing {haz['name']}\n- Never dispose of in regular trash\n- Never attempt DIY removal for large quantities",
            "category": "safety_hazards", "subcategory": "hazardous_material", "tags": [haz["name"], "hazardous", "safety"]})

        # Cross-reference with products
        products_containing = haz["found_in"].split(", ")
        for product in products_containing[:4]:
            entries.append({"id": make_id("safety_product", haz["name"], product), "title": f"Is {product.strip()} safe to handle?",
                "content": f"## {product.strip().title()} — Safety Information\n\n**Contains:** {haz['name']}\n**Risk Level:** ⚠️ Hazardous\n\n**Health Risk:** {haz['health_risk']}\n\n**Safe Handling:**\n- Wear appropriate PPE (gloves, mask if needed)\n- Do not break, crush, or burn\n- Keep away from children and pets\n\n**Disposal:** {haz['disposal']}\n\n**Regulation:** {haz['regulation']}",
                "category": "safety_hazards", "subcategory": "product_safety", "tags": [haz["name"], product.strip(), "safety"]})


def gen_upcycling_ideas(entries: List[Dict]):
    """Generate upcycling project articles."""
    for proj in UPCYCLING_PROJECTS:
        entries.append({"id": make_id("upcycle", proj["title"]), "title": f"Upcycling: {proj['title']}",
            "content": f"## 🔨 {proj['title']}\n\n**Source Material:** {proj['material'].replace('_', ' ').title()}\n**Difficulty:** {proj['difficulty'].title()}\n**Time Required:** {proj['time']}\n**Tools Needed:** {', '.join(proj['tools'])}\n\n**Why Upcycle?**\nUpcycling extends product life and avoids the energy cost of recycling or manufacturing new products. Every item upcycled is one less item in the waste stream.\n\n**Safety Notes:**\n- Wash all materials thoroughly before use\n- Inspect for chips, cracks, or sharp edges\n- Use appropriate tools and wear eye protection when cutting\n- Ensure food-safe materials if used for food storage\n\n**Environmental Impact:**\n- Saves energy vs. recycling (no melting/reprocessing)\n- Extends product lifecycle by 2-10+ years\n- Reduces demand for virgin materials\n- Higher on the waste hierarchy than recycling",
            "category": "upcycling_ideas", "subcategory": proj["difficulty"], "tags": [proj["material"], proj["difficulty"], "upcycling"]})

    # Material-grouped upcycling guides
    materials = set(p["material"] for p in UPCYCLING_PROJECTS)
    for mat in materials:
        projects = [p for p in UPCYCLING_PROJECTS if p["material"] == mat]
        mat_name = mat.replace("_", " ").title()
        project_list = "\n".join(f"- **{p['title']}** ({p['difficulty']}, {p['time']})" for p in projects)
        entries.append({"id": make_id("upcycle_guide", mat), "title": f"Complete Upcycling Guide: {mat_name}",
            "content": f"## Upcycling Guide: {mat_name}\n\n**Available Projects:**\n{project_list}\n\n**Before You Start:**\n1. Clean all materials thoroughly\n2. Inspect for damage\n3. Gather all tools before beginning\n4. Work in a well-ventilated area\n\n**Tip:** Start with beginner projects and work up to intermediate/advanced.",
            "category": "upcycling_ideas", "subcategory": "guide", "tags": [mat, "guide", "upcycling"]})


def gen_sustainability_info(entries: List[Dict]):
    """Generate sustainability topic articles."""
    topic_details = {
        "carbon footprint of common materials": "The carbon footprint measures total greenhouse gas emissions across a product's lifecycle. PET plastic: 2.15 kg CO₂e per kg. Aluminum (virgin): 8.24 kg CO₂e per kg (recycled: 0.46 kg). Glass: 0.85 kg CO₂e per kg. Paper: 1.1 kg CO₂e per kg. Cotton: 5.9 kg CO₂e per kg.",
        "water footprint of textile production": "Global textile industry uses 93 billion cubic meters of water annually. One cotton t-shirt: 2,700 liters. One pair of jeans: 7,500 liters. Dyeing alone accounts for 20% of global industrial water pollution.",
        "circular economy principles": "The circular economy replaces the linear 'take-make-dispose' model with closed loops. Key principles: 1) Design out waste and pollution, 2) Keep products and materials in use, 3) Regenerate natural systems. Pioneered by Ellen MacArthur Foundation.",
        "microplastic pollution in oceans": "8 million tons of plastic enter oceans annually. Microplastics (<5mm) found in 83% of tap water globally. Sources: synthetic textiles (35%), tire wear (28%), city dust (24%), road markings (7%). Found in human blood, lung tissue, and placentas.",
        "fast fashion environmental impact": "Fashion industry produces 10% of global CO₂ emissions. 92 million tons of textile waste annually. Average garment worn 7 times before disposal. 20% of global wastewater from textile dyeing.",
        "composting methods (hot, cold, vermicomposting)": "Hot composting: 55-65°C, kills pathogens, 1-3 months. Cold composting: ambient temperature, 6-12 months, less labor. Vermicomposting: red wiggler worms (Eisenia fetida), room temperature, produces high-quality castings.",
        "biodegradable vs compostable standards": "Biodegradable: no timeframe or conditions specified, largely unregulated. Compostable: must meet ASTM D6400 (US) or EN 13432 (EU) — 90% biodegradation within 180 days at 58°C. BPI certification verifies compliance.",
        "MRF technology and sorting": "Materials Recovery Facilities process 66 million tons annually in the US. Technology: trommel screens (size separation), air classifiers (density), magnets (steel), eddy currents (aluminum), NIR optical sorters (plastics by resin code). Average contamination: 25%.",
        "waste-to-energy technologies": "WtE processes: mass burn incineration (most common), gasification (partial combustion), pyrolysis (no oxygen), plasma arc (highest temperatures). Modern WtE reduces waste volume by 90%, generates electricity. Concerns: air emissions, ash disposal, competes with recycling.",
        "life cycle assessment methodology": "LCA follows ISO 14040/14044. Four phases: 1) Goal and scope definition, 2) Life Cycle Inventory (LCI), 3) Life Cycle Impact Assessment (LCIA), 4) Interpretation. Impact categories: global warming potential, acidification, eutrophication, ozone depletion.",
    }

    for topic in SUSTAINABILITY_TOPICS:
        detail = topic_details.get(topic, f"Comprehensive analysis of {topic} covering environmental impact, current research, policy implications, and practical action steps.")
        entries.append({"id": make_id("sustain", topic), "title": f"Sustainability: {topic.title()}",
            "content": f"## {topic.title()}\n\n{detail}\n\n**Key Takeaway:** Understanding {topic} is essential for making informed sustainability decisions at both individual and organizational levels.\n\n**Sources:** UNEP, EPA, Ellen MacArthur Foundation, World Bank, peer-reviewed literature.",
            "category": "sustainability_info", "subcategory": "topic", "tags": [topic, "sustainability"]})


def gen_policy_regulation(entries: List[Dict]):
    """Generate policy and regulation articles."""
    for region in REGIONS:
        entries.append({"id": make_id("policy", region["name"]), "title": f"Recycling Regulations: {region['name']}",
            "content": f"## {region['name']} — Waste Management Regulations\n\n**Overall Recycling Rate:** {region['recycling_rate']}\n\n**Key Legislation:**\n" + "\n".join(f"- {r}" for r in region['regulations']) + f"\n\n**Notable Features:** {region['notable']}\n\n**Enforcement:** Regulations vary by jurisdiction. Check local municipal guidelines for specific sorting requirements and collection schedules.",
            "category": "policy_regulation", "subcategory": "overview", "tags": [region["name"], "regulation"]})

        # Per-regulation articles
        for reg in region["regulations"]:
            entries.append({"id": make_id("policy_detail", region["name"], reg), "title": f"Regulation: {reg} ({region['name']})",
                "content": f"## {reg}\n\n**Jurisdiction:** {region['name']}\n**Type:** Environmental/Waste Management Regulation\n\n**Summary:** {reg} is a key piece of waste management legislation in {region['name']}. It governs how waste materials are collected, sorted, processed, and disposed of.\n\n**Key Requirements:** Compliance with {reg} may include mandatory sorting, reporting obligations, producer responsibility, and penalties for non-compliance.\n\n**Impact:** This regulation contributes to {region['name']}'s overall recycling rate of {region['recycling_rate']}.",
                "category": "policy_regulation", "subcategory": "detail", "tags": [region["name"], reg, "regulation"]})


def gen_certifications(entries: List[Dict]):
    """Generate certification and standard articles."""
    for cert in CERTIFICATIONS:
        entries.append({"id": make_id("cert", cert["name"]), "title": f"Certification: {cert['name']}",
            "content": f"## {cert['name']}\n\n**Sector:** {cert['sector'].title()}\n\n**Description:** {cert['description']}\n\n**Why It Matters:** Certifications like {cert['name']} provide third-party verification that products or processes meet established environmental standards. For consumers, look for this certification when purchasing {cert['sector']} products. For businesses, obtaining this certification demonstrates environmental commitment and may be required by certain customers or regulations.",
            "category": "organization_search", "subcategory": "certification", "tags": [cert["name"], cert["sector"], "certification"]})


def gen_lifecycle_comparisons(entries: List[Dict]):
    """Generate lifecycle analysis comparison articles."""
    comparisons = [
        ("paper bag", "plastic bag", "Paper bags have 3x higher carbon footprint per use but biodegrade in 1 month. Plastic bags are lighter and less energy-intensive but persist 500+ years. Reusable PP bags break even after 14 uses."),
        ("glass bottle", "plastic bottle", "Glass: 0.85 kg CO₂e, infinitely recyclable, heavier transport. Plastic: 0.2 kg CO₂e, limited recycling cycles, microplastic risk. Glass wins on infinite recyclability; plastic wins on transport emissions."),
        ("aluminum can", "glass bottle", "Aluminum: 95% energy savings when recycled, 60-day can-to-can cycle. Glass: 30% energy savings, heavier. Aluminum is more energy-intensive to produce from virgin ore but far more efficient to recycle."),
        ("cloth diaper", "disposable diaper", "Disposable: 8,000 diapers per child, 500+ years decomp, 3.5 million tons/year in US landfills. Cloth: significant water/energy for washing. LCA depends on washing habits, electricity source, and lifespan."),
        ("electric vehicle", "gasoline vehicle", "EV: higher manufacturing footprint (battery minerals), lower lifetime emissions (0 tailpipe). Break-even: 20,000-50,000 miles depending on grid mix. EVs are definitively better in clean-grid regions."),
        ("hand dryer", "paper towel", "Hand dryers: 9-40g CO₂ per dry (warm air vs jet). Paper towels: 56g CO₂ per dry, create waste. Jet air dryers (Dyson Airblade type) have lowest lifecycle impact."),
        ("tap water", "bottled water", "Bottled water: 3,500x more energy than tap, costs 2,000x more. Produces 1.5 million tons of plastic waste/year. Tap water is monitored by EPA (more regulated than FDA-regulated bottled water)."),
        ("recycled paper", "virgin paper", "Recycled paper: 70% less energy, 50% less water, 74% less air pollution. Can be recycled 5-7 times before fibers are too short. Virgin paper requires logging, pulping (chemical-intensive), and bleaching."),
    ]
    for item_a, item_b, analysis in comparisons:
        entries.append({"id": make_id("lca", item_a, item_b), "title": f"Lifecycle Comparison: {item_a.title()} vs {item_b.title()}",
            "content": f"## {item_a.title()} vs {item_b.title()} — Lifecycle Analysis\n\n{analysis}\n\n**Methodology:** Based on Life Cycle Assessment (ISO 14040/14044) comparing cradle-to-grave environmental impacts including raw material extraction, manufacturing, transportation, use phase, and end-of-life disposal.\n\n**Best Choice:** The most sustainable option is typically the one you already have and reuse consistently. Single-use items of any material are generally worse than reusable alternatives used enough times to amortize their production impact.",
            "category": "lifecycle_analysis", "subcategory": "comparison", "tags": [item_a, item_b, "LCA"]})


def gen_material_cross_references(entries: List[Dict]):
    """Generate cross-reference articles between materials."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    mat_keys = list(all_mats.keys())

    for i, key_a in enumerate(mat_keys):
        for key_b in mat_keys[i+1:i+4]:  # Compare with next 3
            mat_a, mat_b = all_mats[key_a], all_mats[key_b]
            entries.append({"id": make_id("xref", key_a, key_b), "title": f"Comparing {mat_a['name']} and {mat_b['name']}",
                "content": f"## {mat_a['name']} vs {mat_b['name']}\n\n| Property | {mat_a['name']} | {mat_b['name']} |\n|---|---|---|\n| Type | {mat_a['type']} | {mat_b['type']} |\n| Recyclable | {'Yes' if mat_a.get('recyclable') else 'No'} | {'Yes' if mat_b.get('recyclable') else 'No'} |\n| Recycling Rate | {mat_a.get('recycling_rate', 'N/A')} | {mat_b.get('recycling_rate', 'N/A')} |\n| Decomposition | {mat_a.get('decomp_time', 'varies')} | {mat_b.get('decomp_time', 'varies')} |\n| Energy Savings | {mat_a.get('energy_savings', 'N/A')} | {mat_b.get('energy_savings', 'N/A')} |",
                "category": "material_science", "subcategory": "comparison", "tags": [key_a, key_b, "comparison"]})


def gen_faq_entries(entries: List[Dict]):
    """Generate FAQ-style Q&A for common recycling questions."""
    faqs = [
        ("Can I recycle dirty containers?", "No. Containers must be empty and rinsed. Food residue contaminates entire bales of recyclables. A single greasy pizza box can ruin a ton of cardboard. Empty, rinse, and let dry before recycling.", "disposal_guidance"),
        ("Do I need to remove labels from bottles?", "No. Labels are removed during the recycling process through washing and caustic baths. However, remove bottle caps if they are a different material than the bottle (e.g., PP cap on PET bottle).", "disposal_guidance"),
        ("Can I recycle shredded paper?", "It depends. Most curbside programs reject shredded paper because it falls through sorting screens and clogs machinery. Place shredded paper in a paper bag, staple shut, and label 'shredded paper.' Better: compost it.", "disposal_guidance"),
        ("What do the recycling numbers mean?", "The numbers 1-7 inside the chasing arrows are Resin Identification Codes (RIC), NOT recycling symbols. They identify the plastic type: #1 PET, #2 HDPE, #3 PVC, #4 LDPE, #5 PP, #6 PS, #7 Other. Most programs accept only #1 and #2. The symbol does NOT mean the item is recyclable.", "material_science"),
        ("Can I recycle plastic bags in curbside recycling?", "NO. Plastic bags (LDPE #4) tangle in MRF sorting machinery, causing shutdowns. Take clean, dry bags to store drop-off bins (most grocery stores accept them). Or reuse them.", "disposal_guidance"),
        ("Are compostable cups actually composted?", "Rarely. Most labeled 'compostable' require industrial composting (55-60°C for 12+ weeks). They do NOT compost in home bins or landfills. Only 185 industrial composting facilities in the US accept food-service packaging. Most end up in landfill.", "sustainability_info"),
        ("Can I recycle Styrofoam?", "Almost never. Expanded polystyrene (EPS) is 95% air, making collection and transport uneconomical. It is resin code #6. Less than 1% is recycled. Some specialty recyclers accept clean EPS blocks. NEVER put in curbside recycling.", "disposal_guidance"),
        ("Are black plastic containers recyclable?", "No in most areas. NIR optical sorters at MRFs cannot detect black plastic because carbon black pigment absorbs the infrared light. Some facilities use newer detection methods, but most reject it. Use clear or colored containers when possible.", "disposal_guidance"),
        ("Can I recycle broken glass?", "No. Broken glass is a safety hazard for MRF workers and cannot be sorted by color. Wrap broken glass in newspaper, place in a box labeled 'broken glass,' and put in trash. Intact glass bottles and jars CAN be recycled.", "disposal_guidance"),
        ("Is it better to recycle or compost paper?", "Both are good; composting returns nutrients to soil while recycling saves trees. If paper is clean and dry, recycle it (can be recycled 5-7 times). If it's soiled, greasy, or too degraded, compost it.", "sustainability_info"),
        ("Can I recycle pizza boxes?", "Partially. Tear off the greasy bottom and compost it. The clean top can be recycled. If the entire box is greasy, compost the whole thing. Grease contaminates paper recycling.", "disposal_guidance"),
        ("What is single-stream recycling?", "Single-stream means all recyclables (paper, plastic, metal, glass) go in one bin. A MRF sorts them. Pros: higher participation rates. Cons: higher contamination (25-30%), lower material quality. Multi-stream produces cleaner materials but lower participation.", "sustainability_info"),
        ("Can I recycle receipts?", "No. Most thermal paper receipts contain BPA or BPS and cannot be recycled — the chemicals contaminate the paper recycling stream. Decline receipts when possible, or opt for digital receipts.", "disposal_guidance"),
        ("How do I dispose of cooking oil?", "Never pour down the drain — it causes blockages (fatbergs) and sewer overflows. Small amounts: absorb with paper towels and trash. Large amounts: collect in a sealed container and take to a cooking oil recycling center. Some municipalities collect it for biodiesel production.", "disposal_guidance"),
        ("Can I recycle chip bags and candy wrappers?", "No. These are multi-layer films (metalized plastic/aluminum/paper laminate) that cannot be separated for recycling. Some TerraCycle programs accept them. Otherwise: trash.", "disposal_guidance"),
        ("What's the environmental cost of recycling?", "Recycling uses 30-95% less energy than virgin production depending on material. Aluminum: 95% savings. Paper: 60%. Plastic: 75%. Glass: 30%. Even accounting for collection and processing energy, recycling is net positive for all commonly recycled materials.", "lifecycle_analysis"),
        ("Can I put electronics in the recycling bin?", "NEVER. E-waste contains hazardous materials (lead, mercury, cadmium) and lithium batteries that can cause fires. Take to certified e-waste recyclers (R2 or e-Stewards certified). Many retailers (Best Buy, Staples, Apple) offer free e-waste drop-off.", "disposal_guidance"),
        ("Are biodegradable plastics actually good?", "Not necessarily. 'Biodegradable' has no legal timeframe — a plastic that degrades in 1000 years qualifies. Most biodegradable plastics do NOT break down in landfills (anaerobic). They contaminate PET/HDPE recycling. Look for BPI-certified 'compostable' (ASTM D6400) instead, AND verify your local facility accepts them.", "sustainability_info"),
        ("What happens to recycling that's contaminated?", "Contaminated recycling loads are often landfilled entirely — one contaminated item can cause an entire truckload to be rejected. The US contamination rate is 25%. China's 2018 National Sword policy (0.5% contamination limit) caused a global recycling crisis.", "sustainability_info"),
        ("How do I recycle batteries?", "NEVER put batteries in regular trash or recycling. Alkaline (AA, AAA): some areas allow in trash, but recycling is better. Lithium-ion: fire hazard — tape terminals, take to battery recycling (Call2Recycle, Home Depot, Lowes). Button cells: mercury — HHW only. Lead-acid: 99% recycling rate — auto stores accept them.", "safety_hazards"),
    ]
    for i, (q, a, cat) in enumerate(faqs):
        entries.append({"id": make_id("faq", i), "title": q, "content": f"## {q}\n\n{a}",
            "category": cat, "subcategory": "faq", "tags": ["faq"]})


def gen_myth_busting(entries: List[Dict]):
    """Generate myth vs fact articles."""
    myths = [
        ("Myth: The chasing arrows symbol means an item is recyclable", "FACT: The chasing arrows with a number inside are Resin Identification Codes (RIC). They identify the plastic TYPE, not recyclability. Created by the plastics industry in 1988, this symbol has caused massive consumer confusion. Only your local recycling program determines what is actually recyclable in your area."),
        ("Myth: Recycling uses more energy than making new products", "FACT: Recycling saves significant energy for ALL commonly recycled materials. Aluminum: 95% energy savings. Paper: 60%. Plastic: 75%. Glass: 30%. Even accounting for collection, sorting, and processing energy, the net energy savings are substantial."),
        ("Myth: All plastic can be recycled", "FACT: Only plastics #1 (PET) and #2 (HDPE) are widely recyclable. #3 PVC, #6 PS, and #7 Other are almost never recycled. #4 LDPE (bags) requires store drop-off. #5 PP acceptance is growing but still limited. Globally, only 9% of all plastic ever produced has been recycled."),
        ("Myth: It doesn't matter if I put the wrong thing in recycling", "FACT: Contamination is the #1 problem in recycling. One wrong item (garden hose, plastic bag, food waste) can ruin an entire load. The average US MRF contamination rate is 25%. Contaminated loads are often sent to landfill entirely. When in doubt, throw it out."),
        ("Myth: Biodegradable means it will break down quickly", "FACT: 'Biodegradable' has no legal definition for timeframe. A plastic labeled biodegradable might take decades to break down, and in a landfill (anaerobic conditions) it may never fully decompose. Only 'compostable' (ASTM D6400) has defined standards."),
        ("Myth: Recycled products are lower quality", "FACT: Many recycled materials are equal or superior quality. Aluminum and glass can be recycled infinitely with no quality loss. Recycled PET is used in food-grade bottles. Recycled steel is indistinguishable from virgin steel."),
        ("Myth: Landfills are safe because things decompose", "FACT: Modern engineered landfills are designed to PREVENT decomposition (lined, no moisture, no oxygen). A newspaper from the 1960s was found perfectly readable in a landfill dig. Organic waste in landfills produces methane (23x worse than CO₂ for climate)."),
        ("Myth: If I clean and sort perfectly, everything gets recycled", "FACT: Even properly sorted recyclables may not be recycled due to market conditions. When commodity prices drop or markets saturate, recyclables may be stockpiled or landfilled. The US exported 30% of recyclables before China's 2018 National Sword policy."),
        ("Myth: Paper bags are always better than plastic bags", "FACT: Paper bags have 3-4x the carbon footprint of plastic bags (more energy to manufacture, heavier to transport). A paper bag must be reused 3+ times to be environmentally equivalent to a single-use plastic bag. Best option: reusable bags used 14+ times."),
        ("Myth: Wishful recycling (aspirational recycling) helps the environment", "FACT: 'Wish-cycling' — putting non-recyclable items in the bin hoping they'll be recycled — actually HURTS recycling. It contaminates loads, damages equipment, and increases costs. Check your local guidelines and only recycle what is accepted."),
        ("Myth: Composting is only for people with gardens", "FACT: Urban composting options include vermicomposting (worm bins, apartment-friendly), community composting sites, curbside organics collection (growing rapidly), and food waste drop-off. Many cities now mandate organic waste diversion."),
        ("Myth: Recycling solves the plastic pollution problem", "FACT: Recycling is necessary but insufficient. Only 9% of plastic has ever been recycled globally. The real solution requires reducing production, redesigning packaging for recyclability, and building circular economy infrastructure. Reduce > Reuse > Recycle."),
    ]
    for i, (myth, fact) in enumerate(myths):
        entries.append({"id": make_id("myth", i), "title": myth, "content": f"## {myth}\n\n{fact}",
            "category": "sustainability_info", "subcategory": "myth_busting", "tags": ["myth", "fact"]})


def gen_composite_products(entries: List[Dict]):
    """Generate disposal guides for multi-material composite products."""
    composites = [
        {"item": "Coffee Cup (Paper with Plastic Lining)", "materials": ["paper", "PE plastic lining"], "recyclable": False, "disposal": "Most paper coffee cups have a polyethylene (PE) plastic lining that makes them non-recyclable in standard paper recycling. Only specialized facilities can separate the layers. Lids (#6 PS) are also not recyclable. Compostable cups exist but require industrial composting. Best: bring a reusable cup.", "bin": "trash (unless your area has specialized facilities)"},
        {"item": "Juice Box / Carton (Tetra Pak)", "materials": ["paperboard", "polyethylene", "aluminum foil"], "recyclable": True, "disposal": "Tetra Paks are made of 3 layers: 75% paperboard, 20% polyethylene, 5% aluminum. They CAN be recycled in many areas — check locally. Specialized hydrapulper machines separate layers. Rinse, flatten, replace cap. 60% of US has access to carton recycling.", "bin": "recycling (check locally)"},
        {"item": "Chip Bag / Snack Wrapper", "materials": ["metalized film", "plastic layers", "aluminum"], "recyclable": False, "disposal": "Multi-layer flexible packaging (MLP) cannot be separated for recycling. The aluminum, plastic, and adhesive layers are permanently bonded. TerraCycle accepts some brands. Otherwise: trash. This is a major packaging waste problem — 80 billion flexible packages/year in US.", "bin": "trash"},
        {"item": "Blister Pack (Medication)", "materials": ["PVC or PVDC plastic", "aluminum foil"], "recyclable": False, "disposal": "Blister packs combine PVC/PVDC plastic with aluminum backing. Cannot be recycled due to material fusion. Pop pills out first. Some pharmacies have take-back programs. TerraCycle offers blister pack recycling. Most end up in trash.", "bin": "trash"},
        {"item": "Disposable Coffee Pod (K-Cup)", "materials": ["plastic cup", "aluminum lid", "filter", "coffee grounds"], "recyclable": "Partially", "disposal": "Separate components: 1) Peel aluminum lid (recycle), 2) Dump coffee grounds (compost), 3) Remove paper filter (compost), 4) Rinse plastic cup (#5 PP — check local recycling). Some brands offer recyclable or compostable pods.", "bin": "separate components"},
        {"item": "Toothpaste Tube", "materials": ["HDPE or aluminum", "plastic cap"], "recyclable": "Depends", "disposal": "Traditional toothpaste tubes are multi-layer (aluminum + plastic) and NOT recyclable. Newer tubes (Tom's, Colgate) are mono-material HDPE and CAN be recycled. Check the tube for recycling symbol. Roll up to squeeze out remaining paste.", "bin": "check material type"},
        {"item": "Aerosol Can (Spray Paint)", "materials": ["steel or aluminum", "pressurized contents"], "recyclable": True, "disposal": "MUST be completely empty before recycling. Pressurized cans can explode in recycling machinery — extremely dangerous. Spray until empty, let air escape. Remove plastic cap. Most programs accept empty aerosol cans. Never puncture.", "bin": "recycling (when COMPLETELY empty)"},
        {"item": "Wine/Spirit Bottle with Metal Cap", "materials": ["glass", "aluminum or steel cap", "label"], "recyclable": True, "disposal": "Recycle the glass bottle (remove cork if applicable). Metal screw caps can go in recycling. Paper labels are removed during processing. Wine corks: natural cork can be composted or collected by programs like ReCork.", "bin": "recycling bin"},
        {"item": "Pizza Box", "materials": ["corrugated cardboard", "grease", "food residue"], "recyclable": "Partially", "disposal": "Tear the box: clean top → recycle. Greasy bottom → compost (or trash if no composting). If entire box is saturated with grease, compost the whole thing. Cheese and food pieces go in compost. Never put greasy cardboard in paper recycling.", "bin": "split: recycle clean / compost greasy"},
        {"item": "Shoes/Sneakers", "materials": ["rubber sole", "textile upper", "foam", "adhesives", "metal eyelets"], "recyclable": True, "disposal": "Do NOT put in curbside recycling. Options: 1) Donate if wearable (Goodwill, Salvation Army), 2) Nike Reuse-A-Shoe program grinds into Nike Grind material, 3) Soles4 Souls collects shoes for developing countries. Some textile recyclers accept shoes.", "bin": "donation or specialty program"},
        {"item": "Mattress", "materials": ["steel springs", "foam", "cotton", "wood frame", "fabric"], "recyclable": True, "disposal": "Do NOT put at curb in most areas. Mattresses contain 90% recyclable materials: steel springs, wood, cotton, foam. Take to a mattress recycler. Many cities have drop-off programs or curbside pickup for $20-50. Some retailers offer take-back. Never leave in alley/illegal dump.", "bin": "mattress recycler or bulky waste pickup"},
        {"item": "Car Tire", "materials": ["rubber", "steel belts", "textile cords", "carbon black"], "recyclable": True, "disposal": "Never put in trash — illegal in most states. Options: 1) Tire retailers are required to accept old tires, 2) Scrap tire processors shred into crumb rubber (playgrounds, artificial turf), 3) Tire-derived fuel, 4) Civil engineering applications. ~81% of tires are recycled or used for energy.", "bin": "tire retailer or specialty recycler"},
    ]
    for comp in composites:
        entries.append({"id": make_id("composite", comp["item"]), "title": f"How to Recycle: {comp['item']}",
            "content": f"## {comp['item']}\n\n**Materials:** {', '.join(comp['materials'])}\n**Recyclable:** {comp['recyclable']}\n**Bin:** {comp['bin']}\n\n**Disposal Instructions:**\n{comp['disposal']}",
            "category": "disposal_guidance", "subcategory": "composite_product", "tags": comp["materials"] + ["composite"]})


def gen_industry_specific(entries: List[Dict]):
    """Generate industry/business-specific waste management articles."""
    industries = [
        {"name": "Restaurant/Food Service", "waste_types": ["food waste (40-60% of total)", "cardboard packaging", "glass bottles", "aluminum cans", "cooking oil", "polystyrene containers"],
         "best_practices": "Implement source separation (compost, recycling, trash). Partner with composting facility for food waste. Switch to reusable serviceware. Use cooking oil recycling for biodiesel. Train staff on sorting."},
        {"name": "Construction and Demolition", "waste_types": ["concrete (50-60%)", "wood (20-30%)", "drywall", "metal", "roofing", "asbestos (older buildings)"],
         "best_practices": "C&D waste is 40% of US waste stream. Implement deconstruction (vs demolition) to recover materials. Sort on-site: concrete → aggregate, wood → mulch/biomass, metal → scrap. Many jurisdictions require C&D diversion plans."},
        {"name": "Healthcare/Medical", "waste_types": ["regulated medical waste (sharps, pathological)", "pharmaceutical waste", "chemotherapy waste", "general waste (85% of total)"],
         "best_practices": "Segregate at point of generation using color-coded containers (red: biohazard, yellow: chemo, black: RCRA hazardous). 85% is non-hazardous general waste — recycle paper, cardboard, plastics. Use FDA drug take-back programs."},
        {"name": "Office/Commercial", "waste_types": ["paper (60-70%)", "cardboard", "food waste", "plastic containers", "e-waste", "toner cartridges"],
         "best_practices": "Paper is the #1 office waste — implement double-sided printing, digital documents. Recycling bins at every desk. Centralize trash bins (makes people think before tossing). E-waste: use ITAD (IT Asset Disposition) vendors. Toner: manufacturer take-back."},
        {"name": "Manufacturing", "waste_types": ["scrap metal", "plastic offcuts", "solvents", "hazardous chemicals", "packaging waste", "wastewater"],
         "best_practices": "Implement lean manufacturing to reduce waste at source. Industrial symbiosis: one plant's waste becomes another's feedstock. Track waste streams with ISO 14001. Use closed-loop water systems. Segregate hazardous from non-hazardous."},
        {"name": "Retail/E-commerce", "waste_types": ["corrugated cardboard (largest volume)", "shrink wrap/plastic film", "damaged goods", "pallets", "styrofoam packaging"],
         "best_practices": "Cardboard compactors/balers for efficient recycling. Return pallets to suppliers. Film/shrink wrap recycling programs. Donate damaged goods (tax deduction). Switch to recyclable packaging materials. Amazon-style frustration-free packaging."},
        {"name": "Agriculture", "waste_types": ["crop residue", "animal waste", "pesticide containers", "plastic mulch film", "fertilizer bags", "baling twine"],
         "best_practices": "Crop residue: incorporate into soil or biomass energy. Animal waste: anaerobic digestion for biogas/fertilizer. Triple-rinse pesticide containers for recycling. Collect plastic mulch for specialty recyclers. Composting for organic waste."},
        {"name": "Hospitality/Hotels", "waste_types": ["food waste (largest by weight)", "linens/textiles", "toiletry bottles", "cardboard", "glass bottles", "single-use amenities"],
         "best_practices": "Food waste: donation programs (Good Samaritan Act protections), composting. Replace single-use amenities with dispensers (California SB 1162). Linen reuse programs. Centralized recycling stations. Green certifications (Green Key, LEED)."},
    ]
    for ind in industries:
        entries.append({"id": make_id("industry", ind["name"]), "title": f"Waste Management for {ind['name']}",
            "content": f"## {ind['name']} — Waste Management Best Practices\n\n**Key Waste Types:**\n" + "\n".join(f"- {w}" for w in ind["waste_types"]) + f"\n\n**Best Practices:**\n{ind['best_practices']}",
            "category": "sustainability_info", "subcategory": "industry", "tags": [ind["name"], "industry", "business"]})

        # Per waste-type sub-articles
        for wt in ind["waste_types"]:
            entries.append({"id": make_id("industry_waste", ind["name"], wt), "title": f"Managing {wt.split('(')[0].strip()} in {ind['name']}",
                "content": f"## {wt.split('(')[0].strip().title()} — {ind['name']} Sector\n\n**Waste Type:** {wt}\n\n**Context:** In the {ind['name']} sector, {wt.split('(')[0].strip()} is a significant waste stream that requires proper management.\n\n**Key Strategies:**\n- Reduce at source through process optimization\n- Separate from other waste streams at point of generation\n- Identify recycling or recovery options\n- Track quantities for regulatory compliance\n- Train employees on proper handling\n\n**Regulatory Notes:** Check local regulations for specific requirements regarding {wt.split('(')[0].strip()} in {ind['name']} operations.",
                "category": "sustainability_info", "subcategory": "industry_detail", "tags": [ind["name"], wt, "industry"]})


def gen_seasonal_tips(entries: List[Dict]):
    """Generate seasonal waste reduction tips."""
    seasons = [
        {"season": "Holiday Season (November-January)", "tips": [
            "Gift wrap: Use recyclable kraft paper, fabric, or reusable gift bags. Metallic, glittered, or laminated wrapping paper is NOT recyclable.",
            "Christmas trees: Most municipalities offer free curbside pickup or drop-off for chipping into mulch. Remove all decorations, tinsel, and lights first.",
            "Holiday lights: Old string lights contain copper wire — recycle at e-waste facilities or Home Depot holiday lights recycling.",
            "Packaging waste: Flatten all cardboard boxes. Remove styrofoam packing and tape. Bubble wrap and air pillows go to store drop-off with plastic bags.",
            "Food waste: Plan meals carefully — US households waste 30-40% of food. Compost food scraps. Donate excess non-perishables to food banks.",
        ]},
        {"season": "Spring Cleaning (March-May)", "tips": [
            "Declutter responsibly: Donate usable items to charity. Electronics to e-waste. Hazardous items (paint, chemicals) to HHW events.",
            "Garden waste: Start composting yard waste. Many municipalities offer free compost bins. Never bag yard waste in plastic bags — use paper bags or no bags.",
            "Old medications: Use DEA National Prescription Drug Take-Back Day (April). Year-round: pharmacy drop-off boxes. Never flush medications (contaminates water supply).",
            "Mattresses and furniture: Schedule bulky waste pickup or take to recycler. Never dump illegally — fines up to $10,000.",
        ]},
        {"season": "Summer (June-August)", "tips": [
            "BBQ waste: Charcoal ash can be composted (briquette ash may contain additives). Propane tanks: exchange at retailers, never put in trash or recycling.",
            "Pool chemicals: Never pour down drain. Take to HHW facility. Chlorine reacts violently with other chemicals.",
            "Sunscreen bottles: Rinse and recycle HDPE (#2) bottles. Aerosol sunscreen cans must be completely empty before recycling.",
            "Summer produce: Compost watermelon rinds, corn husks, and other produce waste. Perfect season for starting a backyard compost pile.",
        ]},
        {"season": "Back to School (August-September)", "tips": [
            "School supplies: Reuse supplies from last year before buying new. Recycle used notebooks (remove spiral wire). Donate unused supplies.",
            "Electronics: Trade-in or recycle old laptops and tablets through manufacturer programs (Apple, Dell). Best Buy offers free e-waste drop-off.",
            "Clothing: Donate outgrown uniforms and clothing. Textile recycling for worn items. Organize school uniform swaps.",
        ]},
        {"season": "Fall (September-November)", "tips": [
            "Leaf disposal: Compost leaves (brown/carbon material) or use as mulch. Never burn leaves — air pollution and fire risk.",
            "Halloween: Compost carved pumpkins (remove candles). Reuse costumes or donate. Avoid single-use plastic decorations.",
            "Weatherization waste: Old weatherstripping (trash), paint cans (HHW), old windows (C&D recycler or glass recycling).",
        ]},
    ]
    for s in seasons:
        tips_text = "\n".join(f"- {t}" for t in s["tips"])
        entries.append({"id": make_id("seasonal", s["season"]), "title": f"Waste Reduction Tips: {s['season']}",
            "content": f"## {s['season']} — Waste Reduction Guide\n\n{tips_text}",
            "category": "sustainability_info", "subcategory": "seasonal", "tags": [s["season"], "seasonal", "tips"]})

        for i, tip in enumerate(s["tips"]):
            entries.append({"id": make_id("seasonal_tip", s["season"], i), "title": f"{s['season'].split('(')[0].strip()}: {tip[:50]}...",
                "content": f"## {s['season']}\n\n{tip}",
                "category": "sustainability_info", "subcategory": "seasonal_tip", "tags": [s["season"], "seasonal"]})


def gen_did_you_know(entries: List[Dict]):
    """Generate interesting fact entries for engagement."""
    facts = [
        "Americans throw away 25 billion styrofoam cups every year — enough to circle the Earth 436 times.",
        "Recycling one aluminum can saves enough energy to power a TV for 3 hours or a laptop for 5 hours.",
        "The average American generates 4.9 pounds of waste per day — nearly twice the global average.",
        "A glass bottle can be recycled infinitely without any loss in purity or quality — unlike plastic which degrades with each cycle.",
        "It takes 450+ years for a plastic bottle to decompose, but only 60 days for an aluminum can to be recycled and back on the shelf.",
        "The Great Pacific Garbage Patch is twice the size of Texas, containing 1.8 trillion pieces of plastic weighing 80,000 tonnes.",
        "Only 9% of all plastic ever produced has been recycled. 12% has been incinerated. 79% is in landfills or the environment.",
        "Food waste is the single largest category of material placed in US landfills — 24% of all municipal solid waste.",
        "Recycling one ton of paper saves 17 trees, 7,000 gallons of water, 380 gallons of oil, and 3.3 cubic yards of landfill space.",
        "E-waste is the fastest-growing waste stream in the world, growing at 3-5% per year. Only 17% is properly recycled globally.",
        "One quart of motor oil can contaminate up to 250,000 gallons of drinking water.",
        "The energy saved from recycling one glass bottle can power a light bulb for 4 hours.",
        "Cigarette butts are the most littered item in the world — 4.5 trillion per year. The filters are made of cellulose acetate (plastic).",
        "Japan recycles 84% of waste (including energy recovery). Their system has up to 44 sorting categories in some cities.",
        "Sweden recycles 99% of household waste — less than 1% goes to landfill. They even import waste from other countries for energy recovery.",
        "The fashion industry produces more carbon emissions than all international flights and maritime shipping combined.",
        "A single disposable diaper takes 500+ years to decompose. A child uses approximately 8,000 diapers before potty training.",
        "Every ton of recycled steel saves 2,500 pounds of iron ore, 1,400 pounds of coal, and 120 pounds of limestone.",
        "Microplastics have been found in 94% of US tap water samples, 93% of bottled water, and in human blood, lungs, and placentas.",
        "The term 'reduce, reuse, recycle' is ordered by priority — reducing consumption is 10-100x more effective than recycling.",
    ]
    for i, fact in enumerate(facts):
        entries.append({"id": make_id("fact", i), "title": f"Did You Know? #{i+1}",
            "content": f"## 💡 Did You Know?\n\n{fact}\n\n**Why This Matters:** Understanding the scale of waste challenges helps drive individual action and policy change. Every small decision compounds.",
            "category": "sustainability_info", "subcategory": "facts", "tags": ["fact", "did_you_know"]})


def gen_material_comparison_matrix(entries: List[Dict]):
    """Generate comprehensive material comparison entries."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    mat_keys = list(all_mats.keys())

    # All pairwise comparisons
    for i, key_a in enumerate(mat_keys):
        for j, key_b in enumerate(mat_keys):
            if i >= j:
                continue
            mat_a, mat_b = all_mats[key_a], all_mats[key_b]
            entries.append({"id": make_id("compare_full", key_a, key_b),
                "title": f"Comparison: {mat_a['name']} vs {mat_b['name']}",
                "content": f"## {mat_a['name']} vs {mat_b['name']}\n\n| Property | {mat_a['name']} | {mat_b['name']} |\n|---|---|---|\n| Type | {mat_a['type']} | {mat_b['type']} |\n| Recyclable | {'Yes' if mat_a.get('recyclable') else 'No'} | {'Yes' if mat_b.get('recyclable') else 'No'} |\n| Recycling Rate | {mat_a.get('recycling_rate', 'N/A')} | {mat_b.get('recycling_rate', 'N/A')} |\n| Decomposition | {mat_a.get('decomp_time', 'varies')} | {mat_b.get('decomp_time', 'varies')} |\n| Energy Savings | {mat_a.get('energy_savings', 'N/A')} | {mat_b.get('energy_savings', 'N/A')} |\n| Hazards | {'; '.join(mat_a.get('hazards', [])[:2])} | {'; '.join(mat_b.get('hazards', [])[:2])} |",
                "category": "material_science", "subcategory": "comparison", "tags": [key_a, key_b, "comparison"]})


def gen_question_variations(entries: List[Dict]):
    """Generate natural question variations for each material × topic."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    question_templates = [
        ("Can I recycle {product}?", "disposal_guidance", "recyclability",
         lambda m, p: f"## Can I Recycle {p.title()}?\n\n**Material:** {m['name']}\n**Answer:** {'Yes' if m.get('recyclable') else 'No'}.\n\n{'Place in recycling bin. ' + m.get('recycling_rate', '') if m.get('recyclable') else 'This material is not accepted by most curbside programs.'}\n\n**Tips:** {'Rinse container, remove labels/caps if different material.' if m['type'] in ('plastic', 'metal', 'glass') else 'Ensure clean and dry.' if m['type'] == 'paper' else 'Check local guidelines.'}"),
        ("What bin does {product} go in?", "disposal_guidance", "bin_decision",
         lambda m, p: f"## What Bin Does {p.title()} Go In?\n\n**Material:** {m['name']} ({m.get('code', '')})\n**Bin:** {'♻️ Recycling' if m.get('recyclable') else '🟢 Compost' if m['type'] == 'organic' else '⚠️ Hazardous Waste' if m['type'] in ('hazardous', 'chemical') else '🗑️ Trash'}\n\n{'Rinse and place in recycling bin.' if m.get('recyclable') else 'Do NOT place in recycling — will contaminate other materials.'}"),
        ("Is {product} bad for the environment?", "lifecycle_analysis", "impact",
         lambda m, p: f"## Environmental Impact: {p.title()}\n\n**Material:** {m['name']}\n**Decomposition:** {m.get('decomp_time', 'varies')}\n\n**Environmental concerns:** {'; '.join(m.get('hazards', ['none known']))}\n\n**Mitigation:** {'Recycle to save ' + m.get('energy_savings', 'significant') + ' energy vs virgin production.' if m.get('recyclable') else 'Reduce consumption — this material has limited end-of-life options.'}"),
        ("What is {product} made of?", "material_science", "composition",
         lambda m, p: f"## What is {p.title()} Made Of?\n\n**Primary material:** {m['name']}\n**Type:** {m['type']}\n**Chemical formula:** {m.get('chemical_formula', 'varies')}\n**Density:** {m.get('density', 'varies')}\n**Melting point:** {m.get('melting_point', 'varies')}"),
        ("How is {product} recycled?", "material_science", "process",
         lambda m, p: f"## How is {p.title()} Recycled?\n\n**Material:** {m['name']}\n**Current recycling rate:** {m.get('recycling_rate', 'varies')}\n\n**Process:** Collection → Sorting ({'NIR optical sorting' if m['type'] == 'plastic' else 'magnet/eddy current' if m['type'] == 'metal' else 'manual/screening'}) → Cleaning → {'Melting and pelletizing' if m['type'] == 'plastic' else 'Melting and casting' if m['type'] in ('metal', 'glass') else 'Pulping and de-inking' if m['type'] == 'paper' else 'Processing'}\n\n**Recycled into:** {', '.join(m.get('recycled_into', ['varies']))}"),
        ("Is {product} toxic?", "safety_hazards", "toxicity",
         lambda m, p: f"## Is {p.title()} Toxic?\n\n**Material:** {m['name']}\n\n**Known hazards:** {'; '.join(m.get('hazards', ['No significant hazards known']))}\n\n**Safety precautions:** {'Handle with care. Do not burn. Consult HHW facility for disposal.' if any('toxic' in h.lower() or 'hazard' in h.lower() or 'carcino' in h.lower() for h in m.get('hazards', [])) else 'Generally safe to handle. Follow standard recycling procedures.'}"),
        ("Where can I recycle {product}?", "organization_search", "location",
         lambda m, p: f"## Where to Recycle {p.title()}\n\n**Material:** {m['name']}\n\n**Options:**\n1. {'Curbside recycling bin (check local rules)' if m.get('recyclable') else 'Specialty recycler (not curbside)'}\n2. {'Local recycling center/drop-off' if m.get('recyclable') else 'HHW facility' if m['type'] in ('hazardous', 'chemical') else 'Retailer take-back programs'}\n3. Earth911.org search by material and zip code\n4. {'Store drop-off programs' if m['type'] in ('plastic', 'textile') else 'Municipal collection events'}\n\n**Tip:** Call ahead to confirm acceptance."),
        ("What happens to {product} in a landfill?", "sustainability_info", "landfill",
         lambda m, p: f"## What Happens to {p.title()} in a Landfill?\n\n**Decomposition time:** {m.get('decomp_time', 'varies')}\n\n{'In a modern engineered landfill, conditions are anaerobic (no oxygen). Even biodegradable materials decompose slowly. Organic waste produces methane (23x more potent than CO₂). ' if m['type'] in ('paper', 'organic') else 'This material does not biodegrade. It will persist in the landfill indefinitely, potentially leaching chemicals into groundwater. '}\n\n**Better alternative:** {'Recycle — saves ' + m.get('energy_savings', 'significant') + ' energy' if m.get('recyclable') else 'Reduce use and seek specialty recycling options.'}"),
        ("Should I throw away {product}?", "disposal_guidance", "throw_away",
         lambda m, p: f"## Should You Throw Away {p.title()}?\n\n**Material:** {m['name']}\n\n**Short answer:** {'No — recycle it instead!' if m.get('recyclable') else 'No — take to HHW/specialty recycler' if m['type'] in ('hazardous', 'chemical', 'electronics') else 'If no recycling/composting options exist, trash is the last resort.'}\n\n**Better options (in order):**\n1. **Reduce** — avoid buying {p} when alternatives exist\n2. **Reuse** — find a second life for {p}\n3. **Recycle** — {'place in recycling bin' if m.get('recyclable') else 'find specialty recycler'}\n4. **Compost** — {'yes, this is compostable' if m['type'] == 'organic' else 'not applicable for this material'}\n5. **Trash** — last resort only"),
        ("How do I prepare {product} for recycling?", "disposal_guidance", "prep",
         lambda m, p: f"## How to Prepare {p.title()} for Recycling\n\n**Material:** {m['name']}\n\n**Steps:**\n1. {'Empty the container completely' if m['type'] in ('plastic', 'metal', 'glass') else 'Remove any non-' + m['type'] + ' attachments'}\n2. {'Rinse with water (does not need to be spotless)' if m['type'] in ('plastic', 'metal', 'glass') else 'Ensure clean and dry' if m['type'] == 'paper' else 'Check for hazardous components'}\n3. {'Remove caps/lids if different material' if m['type'] == 'plastic' else 'Flatten to save space' if m['type'] == 'paper' else 'Leave intact'}\n4. {'Place loose in recycling bin — do NOT bag recyclables' if m.get('recyclable') else 'Take to designated collection point'}\n\n**Common Mistakes:**\n- Bagging recyclables in plastic bags (clogs sorting machines)\n- Not rinsing containers (food contamination)\n- Including items that look similar but are different materials"),
        ("Can I compost {product}?", "disposal_guidance", "compost",
         lambda m, p: f"## Can You Compost {p.title()}?\n\n**Material:** {m['name']}\n\n**Answer:** {'Yes — excellent compost material!' if m['type'] == 'organic' else 'Yes (brown/carbon material)' if m['type'] == 'paper' and m.get('recyclable') else 'No — this material does not biodegrade in compost.'}\n\n{'Add to compost as ' + ('green/nitrogen material. Chop into small pieces for faster decomposition.' if m['type'] == 'organic' else 'brown/carbon material. Shred for better results.') if m['type'] in ('organic', 'paper') else 'Do NOT put in compost bin. This material will not break down and will contaminate the compost.'}"),
        ("What is the carbon footprint of {product}?", "lifecycle_analysis", "carbon",
         lambda m, p: f"## Carbon Footprint: {p.title()}\n\n**Material:** {m['name']}\n\n**Manufacturing Impact:** Manufacturing {p} from virgin {m['name']} has a significant carbon footprint including raw material extraction, transportation, energy for processing, and packaging.\n\n**Recycling Impact:** {'Recycling saves ' + m.get('energy_savings', 'significant') + ' energy vs virgin production, proportionally reducing CO₂ emissions.' if m.get('recyclable') else 'Limited recycling options mean most impact cannot be recovered.'}\n\n**End-of-Life Impact:** {m.get('decomp_time', 'varies')} decomposition time. {'In landfill, may produce methane.' if m['type'] in ('organic', 'paper') else 'Does not biodegrade — persists indefinitely.'}"),
    ]

    for key, mat in all_mats.items():
        for product in mat.get("products", []):
            for template, category, subcategory, content_fn in question_templates:
                q = template.format(product=product)
                entries.append({"id": make_id("qvar", key, product, subcategory),
                    "title": q, "content": content_fn(mat, product),
                    "category": category, "subcategory": subcategory, "tags": [key, product, subcategory]})


def gen_hazard_region_cross(entries: List[Dict]):
    """Generate hazardous material × region disposal guides."""
    for haz in HAZARDOUS_MATERIALS:
        for region in REGIONS:
            entries.append({"id": make_id("haz_region", haz["name"], region["name"]),
                "title": f"Disposing of {haz['name']} in {region['name']}",
                "content": f"## {haz['name']} Disposal — {region['name']}\n\n**Health Risk:** {haz['health_risk']}\n\n**Found In:** {haz['found_in']}\n\n**Disposal in {region['name']}:** {haz['disposal']}\n\n**Applicable Regulations:**\n- {haz['regulation']}\n- Regional: {', '.join(region['regulations'][:2])}\n\n**Regional Context:** {region['notable']}\n\n⚠️ Always contact local authorities for specific disposal instructions in your municipality.",
                "category": "safety_hazards", "subcategory": "regional_hazard", "tags": [haz["name"], region["name"], "hazardous", "regional"]})


def gen_product_regional_disposal(entries: List[Dict]):
    """Generate product × region disposal articles for the most common products."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    bin_types = {"plastic": "recycling bin", "metal": "recycling bin", "glass": "recycling bin",
                 "paper": "recycling bin", "electronics": "e-waste drop-off", "hazardous": "HHW facility",
                 "organic": "green bin / compost", "textile": "donation bin", "chemical": "HHW facility",
                 "rubber": "tire retailer", "mineral": "C&D recycler", "composite": "trash"}

    for key, mat in all_mats.items():
        products = mat.get("products", [])  # All products per material
        for product in products:
            for region in REGIONS:
                mtype = mat.get("type", "other")
                bin_type = bin_types.get(mtype, "check local")
                entries.append({"id": make_id("prod_region", key, product, region["name"]),
                    "title": f"How to dispose of {product} in {region['name']}",
                    "content": f"## Disposing of {product.title()} — {region['name']}\n\n**Material:** {mat['name']} ({mat.get('code', '')})\n**Bin:** {bin_type}\n**Recyclable:** {'Yes' if mat.get('recyclable') else 'No'}\n\n**{region['name']} Context:**\n- Regional recycling rate: {region['recycling_rate']}\n- Key regulations: {', '.join(region['regulations'][:2])}\n- {region['notable']}\n\n**Instructions:** {'Rinse and place in ' + bin_type if mat.get('recyclable') else 'Check for specialty recyclers or place in regular trash.'}\n\n**Note:** Rules vary between municipalities within {region['name']}. Always check local guidelines.",
                    "category": "disposal_guidance", "subcategory": "product_regional", "tags": [key, product, region["name"]]})


def gen_alternative_uses(entries: List[Dict]):
    """Generate reuse/repurpose ideas for common waste items."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    reuse_ideas = {
        "plastic": ["storage containers", "planters", "organizers", "scoops", "funnel (cut bottle)"],
        "metal": ["candle holder", "cookie cutter", "pen holder", "plant marker", "wind chime"],
        "glass": ["vase", "storage jar", "candle holder", "drinking glass", "terrarium"],
        "paper": ["packing material", "fire starter", "mulch", "papier-mâché", "craft paper"],
        "organic": ["compost", "animal feed", "natural dye", "cleaning agent (citrus)", "garden mulch"],
        "textile": ["cleaning rags", "braided rug", "quilting squares", "pet bed stuffing", "insulation"],
    }

    for key, mat in all_mats.items():
        ideas = reuse_ideas.get(mat["type"], ["creative reuse project", "donation", "material recovery"])
        for product in mat.get("products", []):
            for idea in ideas:
                entries.append({"id": make_id("reuse", key, product, idea),
                    "title": f"Reuse {product} as {idea}",
                    "content": f"## ♻️ Reuse: {product.title()} → {idea.title()}\n\n**Original Item:** {product}\n**Material:** {mat['name']}\n**Reuse Idea:** {idea}\n\n**Why Reuse?** Reusing is higher on the waste hierarchy than recycling. It requires no energy for reprocessing and extends the item's lifecycle.\n\n**How To:**\n1. Clean the {product} thoroughly\n2. Inspect for damage or sharp edges\n3. Prepare for new use as {idea}\n\n**Environmental Benefit:** Avoids the energy cost of recycling ({mat.get('energy_savings', 'significant')} energy savings lost if not recycled) and keeps items out of the waste stream entirely.",
                    "category": "upcycling_ideas", "subcategory": "reuse", "tags": [key, product, idea, "reuse"]})


def gen_common_mistakes(entries: List[Dict]):
    """Generate common recycling mistakes for each material."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    mistake_templates = [
        ("putting_in_bag", "Putting {name} in a plastic bag before recycling", "Never bag recyclables. Plastic bags tangle MRF machinery. Place {name} items loose in the recycling bin."),
        ("not_rinsing", "Not rinsing {name} containers before recycling", "Food-contaminated {name} can contaminate entire bales. Rinse containers — they don't need to be spotless, just free of major food residue."),
        ("wrong_bin", "Putting {name} in the wrong bin", "Placing {name} in the trash when it should be recycled increases contamination and waste. Check your local guidelines."),
        ("wishcycling", "Wish-cycling {name} items that are not accepted", "Not all {name} products are recyclable even though they are made of the same material. Check for resin codes, contamination, and local acceptance. When in doubt, throw it out."),
        ("crushing", "Crushing {name} containers before recycling", "Some MRFs prefer containers not crushed (easier for optical sorters to read). Others prefer flattened. Check local guidelines. Generally: flatten cardboard/cans, leave bottles intact."),
        ("caps_on", "Leaving caps on {name} bottles", "Modern MRFs can handle caps. In many areas, leaving caps ON is now preferred (small items fall through screens). But verify with your local program."),
    ]

    for key, mat in all_mats.items():
        for mistake_id, title_t, content_t in mistake_templates:
            entries.append({"id": make_id("mistake", key, mistake_id),
                "title": f"Common Mistake: {title_t.format(name=mat['name'])}",
                "content": f"## ❌ Common Mistake: {title_t.format(name=mat['name'])}\n\n{content_t.format(name=mat['name'])}\n\n**Material:** {mat['name']}\n**Recyclable:** {'Yes' if mat.get('recyclable') else 'No'}\n**Recycling Rate:** {mat.get('recycling_rate', 'varies')}\n\n**The Correct Way:** Always check your local recycling program's accepted materials list. Programs vary widely between municipalities.",
                "category": "disposal_guidance", "subcategory": "common_mistakes", "tags": [key, "mistakes"]})


def gen_material_alternatives(entries: List[Dict]):
    """Generate sustainable alternative suggestions for each material."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    alternatives = {
        "plastic": ["glass containers", "stainless steel bottles", "beeswax wraps", "silicone bags", "bamboo cutlery", "paper straws"],
        "metal": ["already sustainable — recycle indefinitely", "buy recycled aluminum products"],
        "glass": ["already sustainable — recycle indefinitely", "mason jars for storage"],
        "paper": ["digital documents", "cloth napkins", "reusable notebooks", "bamboo paper"],
        "electronics": ["buy refurbished", "repair instead of replace", "choose modular designs", "lease programs"],
        "hazardous": ["non-toxic alternatives", "natural cleaning products", "LED bulbs (no mercury)", "digital thermometers"],
        "organic": ["meal planning to reduce waste", "freezing leftovers", "community composting"],
        "textile": ["buy secondhand", "clothing swaps", "repair/alter", "choose natural fibers", "capsule wardrobe"],
        "chemical": ["vinegar-based cleaners", "plant-based solvents", "water-based paint"],
        "rubber": ["retreaded tires", "natural rubber products"],
        "mineral": ["reclaimed building materials", "recycled aggregate"],
        "composite": ["mono-material alternatives", "refillable systems"],
    }

    for key, mat in all_mats.items():
        alts = alternatives.get(mat["type"], ["reduce consumption", "choose reusable alternatives"])
        for product in mat.get("products", []):
            alt_list = "\n".join(f"- {a}" for a in alts)
            entries.append({"id": make_id("alt", key, product),
                "title": f"Sustainable Alternatives to {product.title()}",
                "content": f"## Sustainable Alternatives to {product.title()}\n\n**Current Material:** {mat['name']} ({mat['type']})\n**Decomposition:** {mat.get('decomp_time', 'varies')}\n\n**Better Alternatives:**\n{alt_list}\n\n**Why Switch?** {'Recycling rate is only ' + mat.get('recycling_rate', 'low') + ', meaning most ends up in landfills.' if not mat.get('recyclable') or 'low' in str(mat.get('recycling_rate', '')) else 'Even though recyclable, reducing consumption is always better than recycling.'}\n\n**The Waste Hierarchy:**\n1. 🛑 **Refuse** — decline {product} when possible\n2. 🔄 **Reduce** — use less\n3. ♻️ **Reuse** — find second life\n4. 🏭 **Recycle** — last material recovery option\n5. 🗑️ **Rot/Landfill** — absolute last resort",
                "category": "sustainability_info", "subcategory": "alternatives", "tags": [key, product, "alternatives"]})


def gen_environmental_impact_details(entries: List[Dict]):
    """Generate detailed environmental impact articles per material."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    impact_aspects = [
        ("carbon_footprint", "Carbon Footprint", "The total greenhouse gas emissions across the lifecycle of {name}. Manufacturing from virgin materials produces significantly more CO₂ than using recycled feedstock. Recycling {name} saves {energy} energy vs virgin production."),
        ("water_impact", "Water Usage and Pollution", "Manufacturing {name} requires significant water resources. Improper disposal can contaminate waterways. {name} products take {decomp} to decompose, during which they may leach chemicals."),
        ("land_impact", "Land Use and Landfill Impact", "With {decomp} decomposition time, {name} occupies landfill space for extremely long periods. {name} accounts for a portion of the {recycling_rate} recycling rate, meaning most ends up in landfills."),
        ("ocean_impact", "Ocean and Marine Impact", "{name} waste that reaches waterways can harm marine life through entanglement, ingestion, and habitat destruction. " + "Plastics fragment into microplastics that enter the food chain." if True else ""),
        ("energy", "Energy Analysis", "Recycling {name} saves {energy} energy compared to manufacturing from raw materials. The energy embedded in discarded {name} represents a significant economic and environmental loss."),
    ]

    for key, mat in all_mats.items():
        for aspect_id, aspect_title, template in impact_aspects:
            content = template.format(
                name=mat["name"], energy=mat.get("energy_savings", "significant"),
                decomp=mat.get("decomp_time", "varies"), recycling_rate=mat.get("recycling_rate", "varies")
            )
            entries.append({"id": make_id("impact", key, aspect_id),
                "title": f"{aspect_title}: {mat['name']}",
                "content": f"## {aspect_title}: {mat['name']}\n\n{content}\n\n**Key Hazards:** {'; '.join(mat.get('hazards', ['none known']))}\n\n**What You Can Do:** {'Recycle whenever possible. ' if mat.get('recyclable') else 'Minimize usage. '}Reduce consumption, choose reusable alternatives, and ensure proper disposal.",
                "category": "lifecycle_analysis", "subcategory": aspect_id, "tags": [key, aspect_id, "impact"]})



def gen_identification_guides(entries: List[Dict]):
    """Generate how-to-identify guides for each material."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    identification_methods = {
        "plastic": "Look for the resin identification code (number 1-7 inside triangle) on the bottom. {name} is code {code}. Feel: plastics are lighter than glass/metal. Sound: hollow knock. Burn test (caution): plastics melt, don't shatter.",
        "metal": "Use a magnet: steel/iron is magnetic, aluminum and copper are not. Weight: metals are heavy for their size. Sound: metals ring when tapped. Visual: look for rust (steel) or green patina (copper).",
        "glass": "Weight: heavier than plastic. Sound: glass rings when tapped with metal. Visual: transparent or translucent. Feel: smooth, cold to touch. Shatters rather than deforms.",
        "paper": "Tear test: paper tears along fibers. Water test: paper absorbs water (plastic-coated won't). Feel: fibrous texture. Burn test: paper burns to ash, plastics melt.",
        "organic": "Smell: organic waste has distinctive odor. Visual: shows decomposition signs. Feel: soft, moist. Test: will decompose in compost pile.",
        "textile": "Burn test: cotton smells like burning paper, polyester melts and beads, wool smells like burning hair. Label: check care label for fiber content.",
        "electronics": "Contains circuit boards, wires, batteries, or screens. Has power input. May have model/serial numbers. Check manufacturer's recycling program.",
        "hazardous": "Look for warning symbols: skull/crossbones, flame, exclamation mark, corrosion, environment hazard. Check MSDS/SDS sheets. When in doubt, treat as hazardous.",
        "chemical": "Check labels for GHS hazard pictograms. Look for signal words: DANGER or WARNING. Never mix unknown chemicals. Store in original containers.",
        "rubber": "Bounce test: rubber bounces, plastic does not. Smell: distinctive rubber smell. Feel: elastic, returns to shape. Burn test: rubber burns with thick black smoke.",
        "mineral": "Weight: heavy. Scratch test: ceramics scratch glass, glass doesn't scratch ceramics. Sound: dull thud when tapped. Visual: opaque, often glazed.",
        "composite": "Multiple material layers visible at edges. Cannot be easily separated. Check manufacturer info for material composition.",
    }

    for key, mat in all_mats.items():
        method = identification_methods.get(mat["type"], "Check manufacturer labeling and product documentation.")
        method_formatted = method.replace("{name}", mat["name"]).replace("{code}", mat.get("code", "N/A"))
        entries.append({"id": make_id("identify", key),
            "title": f"How to Identify {mat['name']}",
            "content": f"## How to Identify {mat['name']}\n\n**Material Type:** {mat['type']}\n**Resin Code:** {mat.get('code', 'N/A')}\n\n**Identification Methods:**\n{method_formatted}\n\n**Common Products:** {', '.join(mat.get('products', []))}\n\n**Why Identification Matters:** Correct identification ensures proper recycling. Mixing materials (e.g., putting PVC in PET stream) can ruin entire batches of recyclables.",
            "category": "material_science", "subcategory": "identification", "tags": [key, "identification"]})

        # Per-product identification
        for product in mat.get("products", []):
            entries.append({"id": make_id("identify_product", key, product),
                "title": f"How to Identify the Material in {product.title()}",
                "content": f"## Identifying Material: {product.title()}\n\n**Most likely material:** {mat['name']} ({mat.get('code', '')})\n**Material type:** {mat['type']}\n\n**How to check:**\n{method_formatted}\n\n**Recyclable:** {'Yes' if mat.get('recyclable') else 'No'}\n\n**Tip:** When unsure about material type, check the manufacturer's website or packaging for recycling instructions.",
                "category": "material_science", "subcategory": "product_identification", "tags": [key, product, "identification"]})


def gen_region_comparisons(entries: List[Dict]):
    """Generate region-vs-region comparison articles."""
    for i, region_a in enumerate(REGIONS):
        for region_b in REGIONS[i+1:]:
            entries.append({"id": make_id("region_compare", region_a["name"], region_b["name"]),
                "title": f"Recycling Comparison: {region_a['name']} vs {region_b['name']}",
                "content": f"## {region_a['name']} vs {region_b['name']} — Recycling Comparison\n\n| Metric | {region_a['name']} | {region_b['name']} |\n|---|---|---|\n| Recycling Rate | {region_a['recycling_rate']} | {region_b['recycling_rate']} |\n| Key Regulations | {', '.join(region_a['regulations'][:2])} | {', '.join(region_b['regulations'][:2])} |\n\n**{region_a['name']} Notable:** {region_a['notable']}\n\n**{region_b['name']} Notable:** {region_b['notable']}\n\n**Lessons:** Each jurisdiction takes a different approach. {region_a['name']} focuses on {region_a['regulations'][0]}, while {region_b['name']} emphasizes {region_b['regulations'][0]}.",
                "category": "policy_regulation", "subcategory": "comparison", "tags": [region_a["name"], region_b["name"], "comparison"]})


def gen_recycling_program_guides(entries: List[Dict]):
    """Generate guides for specific recycling programs and services."""
    programs = [
        {"name": "Call2Recycle", "type": "batteries", "url": "call2recycle.org", "description": "Free battery recycling at 16,000+ drop-off locations. Accepts rechargeable batteries, cell phones, and household batteries. Drop-off at Home Depot, Lowe's, Best Buy."},
        {"name": "Earth911", "type": "search engine", "url": "earth911.com", "description": "Comprehensive recycling search engine. Enter material + zip code to find local recycling options. Database of 350+ materials and 100,000+ recycling locations."},
        {"name": "TerraCycle", "type": "hard-to-recycle", "url": "terracycle.com", "description": "Recycling programs for items not accepted curbside: chip bags, contact lenses, cigarette butts, beauty products. Free and paid programs available."},
        {"name": "Goodwill / Salvation Army", "type": "reuse/donation", "url": "goodwill.org", "description": "Accept clothing, housewares, electronics, furniture. Items are resold or recycled. Tax-deductible donations. 3,300+ stores nationwide."},
        {"name": "Nike Reuse-A-Shoe", "type": "shoes", "url": "nike.com", "description": "Collects worn-out athletic shoes and grinds into Nike Grind material for playgrounds, tracks, and courts. Drop off at any Nike store."},
        {"name": "PaintCare", "type": "paint", "url": "paintcare.org", "description": "Industry-funded paint recycling in 10 states + DC. Drop off leftover latex and oil-based paint at participating retailers. Good paint is remixed and resold."},
        {"name": "Habitat for Humanity ReStore", "type": "building materials", "url": "habitat.org/restores", "description": "Accepts new and gently used building materials, appliances, and home goods. 900+ locations. Proceeds fund Habitat for Humanity home builds."},
        {"name": "Best Buy E-Waste", "type": "electronics", "url": "bestbuy.com/recycling", "description": "Free e-waste drop-off at all stores. Accepts TVs (up to 32\"), computers, phones, printers, cables, ink cartridges. Haul-away for large items with purchase."},
        {"name": "Staples E-Waste", "type": "electronics", "url": "staples.com/recycling", "description": "Free technology recycling at all stores. Accepts computers, monitors, printers, fax machines, and accessories. Data destruction available."},
        {"name": "Apple Trade-In", "type": "electronics", "url": "apple.com/trade-in", "description": "Trade in Apple devices for credit or free recycling. Apple's Daisy robot disassembles 200 iPhones per hour, recovering 14 materials."},
        {"name": "Soles4Souls", "type": "shoes/clothing", "url": "soles4souls.org", "description": "Distributes shoes and clothing to people in need in 129 countries. Accepts new and gently worn shoes. 57+ million pairs distributed since 2006."},
        {"name": "ReCork", "type": "wine corks", "url": "recork.org", "description": "Collects natural wine corks for recycling into footwear, insulation, and sports equipment. Drop-off at participating wine shops and Whole Foods."},
    ]

    for prog in programs:
        entries.append({"id": make_id("program", prog["name"]),
            "title": f"Recycling Program: {prog['name']}",
            "content": f"## {prog['name']}\n\n**Type:** {prog['type'].title()}\n**Website:** {prog['url']}\n\n**Description:** {prog['description']}\n\n**How to Use:**\n1. Visit {prog['url']} for locations near you\n2. Check accepted materials list\n3. Prepare items according to program guidelines\n4. Drop off during business hours",
            "category": "organization_search", "subcategory": "program", "tags": [prog["name"], prog["type"], "program"]})

        # Cross with materials
        all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
        for key, mat in all_mats.items():
            if mat["type"] in prog["type"] or prog["type"] in ("search engine", "hard-to-recycle"):
                recyclable_str = "Yes" if mat.get("recyclable") else "No"
                rate_str = mat.get("recycling_rate", "varies")
                prog_content = (
                    f"## {mat['name']} via {prog['name']}\n\n"
                    f"**Material:** {mat['name']}\n"
                    f"**Program:** {prog['name']} ({prog['url']})\n\n"
                    f"{prog['description']}\n\n"
                    f"**Specifics:** Recyclable: {recyclable_str}, Rate: {rate_str}")
                entries.append({"id": make_id("program_mat", prog["name"], key),
                    "title": f"Recycling {mat['name']} through {prog['name']}",
                    "content": prog_content,
                    "category": "organization_search", "subcategory": "program_material",
                    "tags": [prog["name"], key, "program"]})


# ═══════════════════════════════════════════════════════════════════════


def gen_product_preparation_tips(entries: List[Dict]):
    """Generate step-by-step preparation tips for recycling specific products × specific regions."""
    all_mats = {**MATERIALS, **MATERIAL_EXTRAS}
    prep_steps = {
        "plastic": ["1. Empty the container completely", "2. Rinse with water (not spotless)", "3. Check resin code on bottom", "4. Remove caps if different material", "5. Place LOOSE in recycling bin (no bags)"],
        "metal": ["1. Empty the container", "2. Rinse to remove food", "3. Can be crushed to save space", "4. Leave label on (removed during processing)", "5. Place in recycling bin"],
        "glass": ["1. Empty and rinse", "2. Remove metal lids/caps (recycle separately)", "3. Do NOT break (safety hazard at MRF)", "4. Separate by color if required", "5. Place in glass recycling"],
        "paper": ["1. Remove any plastic windows/liners", "2. Remove tape and staples (if easy)", "3. Flatten boxes", "4. Keep dry (wet paper = contamination)", "5. Bundle or place loose in bin"],
        "electronics": ["1. Back up data if applicable", "2. Factory reset device", "3. Remove batteries (recycle separately)", "4. Bring to certified e-waste recycler", "5. Get certificate of recycling for data security"],
        "hazardous": ["1. Keep in original container", "2. Do NOT mix with other waste", "3. Wear gloves", "4. Transport upright in sealed bag", "5. Bring to HHW facility"],
    }

    for key, mat in all_mats.items():
        mtype = mat.get("type", "other")
        steps = prep_steps.get(mtype, ["1. Check local guidelines", "2. Separate from other waste", "3. Transport safely"])
        steps_text = "\n".join(steps)

        for product in mat.get("products", []):
            for region in REGIONS:  # All regions
                entries.append({"id": make_id("prep", key, product, region["name"]),
                    "title": f"Preparing {product} for recycling in {region['name']}",
                    "content": f"## Preparing {product.title()} for Recycling — {region['name']}\n\n**Material:** {mat['name']}\n**Region:** {region['name']} (recycling rate: {region['recycling_rate']})\n\n**Steps:**\n{steps_text}\n\n**Regional Note:** {region['notable']}\n\n**Key Regulations:** {', '.join(region['regulations'][:2])}",
                    "category": "disposal_guidance", "subcategory": "preparation_regional", "tags": [key, product, region["name"], "preparation"]})


def gen_material_substitution_guides(entries: List[Dict]):
    """Generate material substitution recommendations for industrial applications."""
    substitutions = [
        ("PVC", "HDPE", "plumbing", "HDPE pipe is safer (no chlorine, no dioxins) and more widely recyclable. HDPE has better chemical resistance for most residential applications."),
        ("PS", "PP", "food containers", "PP is food-safe, microwave-safe, and recyclable. PS (#6) is rarely recycled and styrene is a possible carcinogen."),
        ("PET", "glass", "beverage containers", "Glass is infinitely recyclable with no quality loss. PET degrades with each cycle. However, glass is heavier (higher transport emissions)."),
        ("LDPE", "paper", "bags", "Paper bags are compostable but have 3x carbon footprint. Best: reusable bags used 14+ times."),
        ("PS", "PLA", "disposable cups", "PLA (polylactic acid) is plant-based and industrially compostable. But PLA contaminates PET recycling streams. Only viable where industrial composting exists."),
        ("PVC", "PP", "packaging film", "PP is recyclable and does not contain chlorine. PVC releases dioxins when burned and contains phthalate plasticizers."),
        ("aluminum", "steel", "cans", "Both are highly recyclable. Aluminum saves 95% energy when recycled (vs 74% for steel). Steel is cheaper and magnetic (easier to sort)."),
        ("virgin plastic", "recycled plastic", "packaging", "Post-consumer recycled (PCR) content reduces virgin material demand. Many brands target 25-50% PCR by 2025."),
        ("bleached paper", "unbleached paper", "packaging", "Unbleached (kraft) paper avoids chlorine bleaching chemicals and is more easily recyclable. Slightly lower aesthetic appeal."),
        ("conventional cotton", "organic cotton", "textiles", "Organic cotton uses 62% less energy and 88% less water. No pesticides or synthetic fertilizers. More expensive but significantly lower environmental impact."),
        ("synthetic fiber", "recycled polyester", "textiles", "Recycled polyester (rPET) from bottles uses 59% less energy than virgin polyester. Same performance properties."),
        ("single-use plastic", "compostable packaging", "food service", "BPI-certified compostable packaging works only where industrial composting exists. Otherwise ends up in landfill like plastic."),
    ]
    for orig, alt, application, reasoning in substitutions:
        entries.append({"id": make_id("sub", orig, alt, application),
            "title": f"Material Substitution: {orig} → {alt} for {application}",
            "content": f"## Switching from {orig} to {alt} in {application.title()}\n\n**Current Material:** {orig}\n**Recommended Alternative:** {alt}\n**Application:** {application}\n\n**Why Switch?**\n{reasoning}\n\n**Considerations:**\n- Cost: evaluate total cost of ownership including disposal\n- Performance: verify alternative meets application requirements\n- Supply chain: ensure reliable supply of alternative material\n- Certification: check if new material meets regulatory requirements\n\n**Environmental Benefit:** Material substitution is a key strategy in the circular economy and sustainable procurement.",
            "category": "sustainability_info", "subcategory": "substitution", "tags": [orig, alt, application, "substitution"]})

        # Regional variants
        for region in REGIONS:
            entries.append({"id": make_id("sub_region", orig, alt, region["name"]),
                "title": f"{orig} → {alt} substitution in {region['name']}",
                "content": f"## {orig} → {alt} Substitution — {region['name']}\n\n**Application:** {application}\n**Region:** {region['name']}\n\n{reasoning}\n\n**Regional Context:**\n- Recycling rate: {region['recycling_rate']}\n- Regulations: {', '.join(region['regulations'][:2])}\n- {region['notable']}\n\n**Regulatory Support:** Check if {region['name']} offers incentives for material substitution or penalizes use of {orig}.",
                "category": "policy_regulation", "subcategory": "substitution_regional", "tags": [orig, alt, region["name"], "substitution"]})


# MAIN — orchestrate all generators
# ═══════════════════════════════════════════════════════════════════════

def main():
    entries: List[Dict] = []

    print("Generating knowledge corpus...")
    gen_material_science(entries)
    print(f"  material_science: {len(entries)} entries")

    n = len(entries)
    gen_disposal_guidance(entries)
    print(f"  disposal_guidance: {len(entries) - n} entries")

    n = len(entries)
    gen_safety_hazards(entries)
    print(f"  safety_hazards: {len(entries) - n} entries")

    n = len(entries)
    gen_upcycling_ideas(entries)
    print(f"  upcycling_ideas: {len(entries) - n} entries")

    n = len(entries)
    gen_sustainability_info(entries)
    print(f"  sustainability_info: {len(entries) - n} entries")

    n = len(entries)
    gen_policy_regulation(entries)
    print(f"  policy_regulation: {len(entries) - n} entries")

    n = len(entries)
    gen_certifications(entries)
    print(f"  certifications: {len(entries) - n} entries")

    n = len(entries)
    gen_lifecycle_comparisons(entries)
    print(f"  lifecycle_analysis: {len(entries) - n} entries")

    n = len(entries)
    gen_material_cross_references(entries)
    print(f"  cross_references: {len(entries) - n} entries")

    n = len(entries)
    gen_faq_entries(entries)
    print(f"  faq_entries: {len(entries) - n} entries")

    n = len(entries)
    gen_myth_busting(entries)
    print(f"  myth_busting: {len(entries) - n} entries")

    n = len(entries)
    gen_composite_products(entries)
    print(f"  composite_products: {len(entries) - n} entries")

    n = len(entries)
    gen_industry_specific(entries)
    print(f"  industry_specific: {len(entries) - n} entries")

    n = len(entries)
    gen_seasonal_tips(entries)
    print(f"  seasonal_tips: {len(entries) - n} entries")

    n = len(entries)
    gen_did_you_know(entries)
    print(f"  did_you_know: {len(entries) - n} entries")

    n = len(entries)
    gen_material_comparison_matrix(entries)
    print(f"  comparison_matrix: {len(entries) - n} entries")

    n = len(entries)
    gen_question_variations(entries)
    print(f"  question_variations: {len(entries) - n} entries")

    n = len(entries)
    gen_hazard_region_cross(entries)
    print(f"  hazard_region_cross: {len(entries) - n} entries")

    n = len(entries)
    gen_product_regional_disposal(entries)
    print(f"  product_regional: {len(entries) - n} entries")

    n = len(entries)
    gen_alternative_uses(entries)
    print(f"  alternative_uses: {len(entries) - n} entries")

    n = len(entries)
    gen_environmental_impact_details(entries)
    print(f"  environmental_impact: {len(entries) - n} entries")

    n = len(entries)
    gen_common_mistakes(entries)
    print(f"  common_mistakes: {len(entries) - n} entries")

    n = len(entries)
    gen_material_alternatives(entries)
    print(f"  material_alternatives: {len(entries) - n} entries")

    n = len(entries)
    gen_identification_guides(entries)
    print(f"  identification_guides: {len(entries) - n} entries")

    n = len(entries)
    gen_region_comparisons(entries)
    print(f"  region_comparisons: {len(entries) - n} entries")

    n = len(entries)
    gen_recycling_program_guides(entries)
    print(f"  program_guides: {len(entries) - n} entries")

    n = len(entries)
    gen_product_preparation_tips(entries)
    print(f"  preparation_tips: {len(entries) - n} entries")

    n = len(entries)
    gen_material_substitution_guides(entries)
    print(f"  substitution_guides: {len(entries) - n} entries")

    # Deduplicate by ID
    seen_ids = set()
    unique = []
    for entry in entries:
        if entry["id"] not in seen_ids:
            seen_ids.add(entry["id"])
            unique.append(entry)
    entries = unique

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Report
    cats = {}
    total_chars = 0
    for e in entries:
        cat = e["category"]
        cats[cat] = cats.get(cat, 0) + 1
        total_chars += len(e["content"])

    print(f"\n{'='*50}")
    print(f"KNOWLEDGE CORPUS GENERATED")
    print(f"{'='*50}")
    print(f"Total entries: {len(entries)}")
    print(f"Total content: {total_chars:,} characters ({total_chars // 4:,} tokens approx)")
    print(f"\nBy category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
