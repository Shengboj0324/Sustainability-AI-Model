#!/usr/bin/env python3
"""
SFT Dataset Generator — Transforms knowledge corpus into 2,000+ training examples.

Reads: data/knowledge_corpus/sustainability_knowledge.jsonl (6,832 entries)
Outputs: data/processed/llm_sft/generated_sft_train.jsonl (1,800+ examples)
         data/processed/llm_sft/generated_sft_val.jsonl   (200+ examples)

Format: {"messages": [system, user, assistant], "category": str, "reasoning_type": str}

Features:
  - Chain-of-thought reasoning traces
  - Multiple question styles per knowledge entry
  - Safety-critical entries get extra emphasis
  - Expert system prompts matching production deployment
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

random.seed(42)

INPUT_PATH = Path(__file__).parent.parent / "data" / "knowledge_corpus" / "sustainability_knowledge.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed" / "llm_sft"

SYSTEM_PROMPT = """You are ReLEAF AI, a Senior Environmental Scientist specializing in waste management, recycling, and sustainability. You provide accurate, safety-conscious advice grounded in verified domain knowledge.

RULES:
1. Always identify materials precisely (resin codes, chemical names)
2. Flag ALL safety hazards prominently with ⚠️
3. Cite regulations when relevant (EPA, OSHA, EU directives)
4. Provide step-by-step disposal instructions
5. Note that rules vary by municipality — advise checking local guidelines
6. Use the waste hierarchy: Refuse > Reduce > Reuse > Recycle > Rot > Landfill
7. Never recommend unsafe disposal (burning plastics, mixing chemicals, etc.)"""


# ═══════════════════════════════════════════════════════════════════════
# QUESTION GENERATORS — create natural questions from knowledge entries
# ═══════════════════════════════════════════════════════════════════════

def gen_questions_from_entry(entry: Dict) -> List[Tuple[str, str, str]]:
    """
    Generate (question, reasoning_type, thinking_trace) tuples from a knowledge entry.
    Uses the entry's unique title + tags to create specific, non-duplicate questions.
    """
    title = entry.get("title", "")
    content = entry.get("content", "")
    category = entry.get("category", "")
    subcategory = entry.get("subcategory", "")
    tags = entry.get("tags", [])
    entry_id = entry.get("id", "")

    results = []

    # Always generate a title-based question (guaranteed unique per entry)
    item = _extract_key_item(tags)
    # Use hash to deterministically pick template variants
    seed = int(hashlib.md5(entry_id.encode()).hexdigest()[:8], 16)

    if category == "disposal_guidance":
        results.extend(_gen_disposal_questions(title, content, tags, seed))
    elif category == "material_science":
        results.extend(_gen_material_questions(title, content, tags, seed))
    elif category == "safety_hazards":
        results.extend(_gen_safety_questions(title, content, tags, seed))
    elif category == "upcycling_ideas":
        results.extend(_gen_upcycling_questions(title, content, tags, seed))
    elif category == "sustainability_info":
        results.extend(_gen_sustainability_questions(title, content, tags, seed))
    elif category == "policy_regulation":
        results.extend(_gen_policy_questions(title, content, tags, seed))
    elif category == "lifecycle_analysis":
        results.extend(_gen_lifecycle_questions(title, content, tags, seed))
    elif category == "organization_search":
        results.extend(_gen_org_questions(title, content, tags, seed))

    return results


def _extract_key_item(tags: List[str]) -> str:
    """Extract most likely item/material name from tags."""
    skip = {"disposal", "safety", "comparison", "faq", "regional", "identification",
            "upcycling", "guide", "facts", "myth", "fact", "seasonal", "industry",
            "alternatives", "substitution", "preparation", "program", "mistakes",
            "reuse", "hazardous", "impact", "did_you_know"}
    for tag in tags:
        if tag.lower() not in skip and len(tag) > 2:
            return tag
    return "this item"


def _pick(templates, seed):
    """Pick a template deterministically based on seed."""
    return templates[seed % len(templates)]


def _gen_disposal_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 and tags[1] not in ("disposal",) else item
    templates = [
        (f"How do I dispose of {item}?", "step_by_step"),
        (f"Can I recycle {item}?", "chain_of_thought"),
        (f"What bin does {item} go in?", "direct_answer"),
        (f"I have some {item} to throw away, what should I do?", "step_by_step"),
        (f"Where does {item} belong - recycling, trash, or compost?", "chain_of_thought"),
        (f"Is {item2} recyclable in my area?", "direct_answer"),
        (f"What's the right way to get rid of {item}?", "step_by_step"),
        (f"Should I put {item2} in the recycling bin?", "chain_of_thought"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nThe user is asking about {item} disposal.\n1. Identify the material type\n2. Check recyclability\n3. Determine correct bin\n4. Note any safety concerns\n5. Provide regional caveats\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_material_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"What is {item} made of?", "factual"),
        (f"Tell me about {item} as a material", "comprehensive"),
        (f"What are the properties of {item}?", "technical"),
        (f"What type of material is {item2}?", "factual"),
        (f"How is {item} manufactured and what's its composition?", "comprehensive"),
        (f"Explain the chemical properties of {item}", "technical"),
        (f"What's the difference between {item} and similar materials?", "comparative"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nMaterial science question about {item}.\n1. Identify chemical composition\n2. List physical properties\n3. Note recyclability\n4. Mention environmental impact\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_safety_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"Is {item} dangerous? How do I handle it safely?", "safety_assessment"),
        (f"What are the health risks of {item}?", "risk_analysis"),
        (f"How do I safely dispose of {item}?", "safety_procedure"),
        (f"Can {item2} harm my health?", "risk_analysis"),
        (f"What safety precautions should I take with {item}?", "safety_assessment"),
        (f"Is it safe to handle {item2} without gloves?", "safety_procedure"),
        (f"What PPE do I need when dealing with {item}?", "safety_assessment"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\n⚠️ SAFETY-CRITICAL QUERY about {item}.\n1. Identify hazard type and severity\n2. List health risks\n3. Specify PPE requirements\n4. Provide safe disposal procedure\n5. Reference applicable regulations\n6. Include emergency contact info\n</thinking>\n\n"
    return [(q, rt, thinking)]



def _gen_upcycling_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"What can I make from old {item}?", "creative"),
        (f"DIY projects using {item}", "creative"),
        (f"How can I upcycle {item} instead of throwing it away?", "creative"),
        (f"Give me craft ideas for reusing {item2}", "creative"),
        (f"I have leftover {item}, any creative uses?", "creative"),
        (f"Fun weekend project ideas using {item2}", "creative"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nUpcycling request for {item}.\n1. Identify material suitability\n2. Suggest projects by difficulty level\n3. List tools needed\n4. Note safety precautions\n5. Explain environmental benefit vs disposal\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_sustainability_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"Tell me about {item} and sustainability", "educational"),
        (f"What is the environmental impact of {item}?", "analysis"),
        (f"Why does {item} matter for the environment?", "educational"),
        (f"How does {item2} affect our planet?", "analysis"),
        (f"What should I know about {item} from a sustainability perspective?", "educational"),
        (f"Is {item2} an environmental concern? Why?", "analysis"),
        (f"Explain the sustainability implications of {item}", "educational"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nSustainability education question about {item}.\n1. Define the concept\n2. Explain environmental significance\n3. Provide data/statistics\n4. Suggest actionable steps\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_policy_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"What are the regulations for {item}?", "regulatory"),
        (f"Tell me about {item} recycling policy", "regulatory"),
        (f"What laws apply to {item} waste management?", "regulatory"),
        (f"Are there any legal requirements for handling {item2}?", "regulatory"),
        (f"What does the law say about disposing of {item}?", "regulatory"),
        (f"Recycling compliance rules for {item2}", "regulatory"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nPolicy/regulation question about {item}.\n1. Identify applicable jurisdiction\n2. List key regulations\n3. Explain requirements\n4. Note enforcement mechanisms\n5. Mention recent changes\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_lifecycle_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"What is the carbon footprint of {item}?", "quantitative"),
        (f"Compare the environmental impact of {item}", "comparative"),
        (f"Is {item} sustainable? What does the data say?", "analysis"),
        (f"How much energy does it take to produce {item2}?", "quantitative"),
        (f"What's the life cycle impact of {item}?", "analysis"),
        (f"Environmental cost analysis for {item2}", "quantitative"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nLifecycle analysis question about {item}.\n1. Define scope (cradle-to-grave)\n2. Quantify energy, water, CO2 impacts\n3. Compare alternatives\n4. Note methodology (ISO 14040)\n5. Provide actionable recommendation\n</thinking>\n\n"
    return [(q, rt, thinking)]


def _gen_org_questions(title, content, tags, seed) -> List[Tuple[str, str, str]]:
    item = _extract_key_item(tags)
    item2 = tags[1] if len(tags) > 1 else item
    templates = [
        (f"Where can I recycle {item}?", "search"),
        (f"Find recycling programs for {item}", "search"),
        (f"What organizations handle {item} recycling?", "informational"),
        (f"Is there a drop-off location for {item2} near me?", "search"),
        (f"Which companies recycle {item}?", "informational"),
        (f"How do I find a recycler for {item2}?", "search"),
    ]
    q, rt = _pick(templates, seed)
    thinking = f"<thinking>\nOrganization/program search for {item}.\n1. Identify relevant recycling programs\n2. List drop-off locations\n3. Note any certifications\n4. Provide contact information\n</thinking>\n\n"
    return [(q, rt, thinking)]


# ═══════════════════════════════════════════════════════════════════════
# ANSWER FORMATTER — converts knowledge content into training-quality answers
# ═══════════════════════════════════════════════════════════════════════

def format_answer(content: str, category: str, thinking: str) -> str:
    """Format a knowledge entry's content as a high-quality assistant response."""
    # Add thinking trace for chain-of-thought training
    answer = thinking + content

    # Add safety footer for hazardous content
    if category == "safety_hazards":
        if "Poison Control" not in answer:
            answer += "\n\n⚠️ **If exposed to hazardous materials:**\n- Call Poison Control: 1-800-222-1222 (US)\n- Call 911 for emergencies\n- Move to fresh air if inhaled"

    # Add locality disclaimer for disposal guidance
    if category == "disposal_guidance":
        if "local guidelines" not in answer.lower():
            answer += "\n\n*⚠️ Recycling rules vary by municipality. Always check your local guidelines.*"

    return answer


def build_sft_example(question: str, answer: str, category: str, reasoning_type: str) -> Dict:
    """Build a single SFT training example in the expected messages format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "category": category,
        "reasoning_type": reasoning_type,
    }


# ═══════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL EXAMPLES — high-quality hand-crafted edge cases
# ═══════════════════════════════════════════════════════════════════════

SUPPLEMENTAL_EXAMPLES = [
    # Safety-critical examples the model MUST learn
    build_sft_example(
        "Can I burn styrofoam to get rid of it?",
        "<thinking>\n⚠️ SAFETY-CRITICAL: User is proposing to burn polystyrene.\n1. Burning PS releases toxic styrene gas, carbon monoxide, and soot\n2. Styrene is IARC Group 2B possible carcinogen\n3. Must strongly advise against this\n4. Provide safe alternatives\n</thinking>\n\n## ⚠️ NEVER Burn Styrofoam\n\n**Burning polystyrene (Styrofoam) is extremely dangerous.**\n\n**What happens when you burn it:**\n- Releases **styrene gas** — a neurotoxin and possible carcinogen (IARC Group 2B)\n- Produces **carbon monoxide** — odorless, deadly gas\n- Creates **thick black toxic smoke** containing PAHs (polycyclic aromatic hydrocarbons)\n- Can cause respiratory damage, headaches, and long-term health effects\n\n**It is also illegal** in most jurisdictions to burn waste materials outdoors.\n\n**Safe disposal options:**\n1. Check if your area has EPS (expanded polystyrene) drop-off recycling\n2. Some shipping stores accept clean packing peanuts\n3. If no recycling available: bag and place in regular trash\n4. Never burn, never put in recycling bin\n\n**Environmental note:** Styrofoam is resin code #6 (PS). Less than 1% is recycled. This is a packaging problem that requires industry-level solutions.",
        "safety_hazards", "safety_assessment"
    ),
    build_sft_example(
        "My kid swallowed a button battery, what should I do?",
        "<thinking>\n🚨 MEDICAL EMERGENCY — button battery ingestion.\n1. This is a life-threatening emergency\n2. Button batteries can cause fatal internal burns within 2 hours\n3. Must direct to emergency services immediately\n4. Provide specific first aid guidance\n</thinking>\n\n## 🚨 MEDICAL EMERGENCY — Call 911 Immediately\n\n**Button battery ingestion is a life-threatening emergency.** Do NOT wait for symptoms.\n\n**Immediate actions:**\n1. **Call 911** or go to the nearest Emergency Room immediately\n2. **Call the National Battery Ingestion Hotline: 1-800-498-8666** (US, 24/7)\n3. **Do NOT induce vomiting** — this can cause more damage\n4. **Do NOT let the child eat or drink** until evaluated by a doctor\n5. If the child is over 12 months old, give **2 teaspoons of honey** every 10 minutes on the way to the ER (honey slows the burn — peer-reviewed evidence)\n\n**Why this is critical:**\n- Button batteries can create an electrical circuit in the esophagus\n- This generates sodium hydroxide (lye), causing severe chemical burns\n- Tissue damage can begin in **as little as 15 minutes**\n- Can be fatal within **2 hours** if lodged in the esophagus\n- Over 2,500 button battery ingestions/year in the US, with 10-20 deaths\n\n**Prevention:** Keep button batteries locked away. Tape battery compartments shut on devices. Dispose of used batteries immediately in a sealed container out of reach.",
        "safety_hazards", "emergency_response"
    ),
    build_sft_example(
        "What do the recycling numbers on plastic mean?",
        "<thinking>\nCommon misconception question — the chasing arrows are NOT recycling symbols.\n1. Explain what RIC codes actually mean\n2. List all 7 codes with recyclability\n3. Correct the misconception that the symbol = recyclable\n4. Provide practical guidance\n</thinking>\n\n## Resin Identification Codes (RIC) — What the Numbers Actually Mean\n\n**Important:** The chasing arrows symbol with a number does **NOT** mean the item is recyclable. It identifies the **type of plastic resin** used.\n\n| Code | Resin | Common Products | Widely Recyclable? |\n|------|-------|----------------|--------------------|\n| #1 | PET (Polyethylene Terephthalate) | Water bottles, soda bottles | ✅ Yes |\n| #2 | HDPE (High-Density Polyethylene) | Milk jugs, detergent bottles | ✅ Yes |\n| #3 | PVC (Polyvinyl Chloride) | Pipes, vinyl flooring | ❌ No (contains chlorine) |\n| #4 | LDPE (Low-Density Polyethylene) | Grocery bags, shrink wrap | ⚠️ Store drop-off only |\n| #5 | PP (Polypropylene) | Yogurt cups, bottle caps | ⚠️ Growing acceptance |\n| #6 | PS (Polystyrene) | Styrofoam, disposable cups | ❌ No (<1% recycled) |\n| #7 | Other (PC, PLA, ABS, etc.) | Misc items | ❌ No (mixed materials) |\n\n**Key takeaway:** Only #1 and #2 are widely accepted in curbside recycling. The symbol was created by the plastics industry in 1988 and has caused massive consumer confusion.\n\n**What to do:** Check your local recycling program's accepted materials list — don't rely on the symbol alone.",
        "material_science", "chain_of_thought"
    ),
    # Multi-turn conversation example
    build_sft_example(
        "Is it better to recycle or compost paper?",
        "<thinking>\nComparative question requiring nuanced analysis.\n1. Both are valid end-of-life options\n2. Depends on paper condition (clean vs contaminated)\n3. Different environmental benefits for each\n4. Provide decision framework\n</thinking>\n\n## Recycling vs. Composting Paper — It Depends on Condition\n\n**Decision Framework:**\n\n| Condition | Best Option | Why |\n|-----------|-------------|-----|\n| Clean, dry paper | ♻️ **Recycle** | Saves trees, water, and energy |\n| Greasy/food-soiled | 🟢 **Compost** | Grease contaminates paper recycling |\n| Shredded paper | 🟢 **Compost** | Falls through MRF screens |\n| Wet/moldy | 🟢 **Compost** | Too degraded for fiber recovery |\n| Wax/plastic-coated | 🗑️ **Trash** | Neither recyclable nor compostable |\n\n**Recycling benefits:** Paper can be recycled 5-7 times before fibers become too short. Each cycle saves 17 trees, 7,000 gallons of water, and 4,100 kWh per ton.\n\n**Composting benefits:** Returns carbon and nutrients to soil. Paper is excellent \"brown\" (carbon) material for compost balance. Decomposes in 2-6 weeks.\n\n**The best answer:** Clean paper → recycle first (higher value recovery). After 5-7 cycles when fibers are too short → compost. Contaminated paper → compost directly.\n\n*Both options are far better than landfill, where paper produces methane (23x more potent than CO₂).*",
        "sustainability_info", "comparative_analysis"
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# MAIN — orchestrate generation
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Load knowledge corpus
    print("Loading knowledge corpus...")
    with open(INPUT_PATH) as f:
        corpus = [json.loads(line) for line in f]
    print(f"  Loaded {len(corpus)} knowledge entries")

    # Generate SFT examples from corpus
    print("Generating SFT examples...")
    examples = []
    skipped = 0

    seen_q_set = set()
    for entry in corpus:
        questions = gen_questions_from_entry(entry)
        if not questions:
            skipped += 1
            continue

        content = entry.get("content", "")
        category = entry.get("category", "")
        title = entry.get("title", "")

        for question, reasoning_type, thinking in questions:
            # If question would be duplicate, create a title-based variant
            if question in seen_q_set and title:
                # Convert title to a natural question
                if title.startswith("How") or title.startswith("What") or title.startswith("Can") or title.startswith("Is") or title.startswith("Where") or title.startswith("Should"):
                    question = title if title.endswith("?") else title + "?"
                else:
                    question = f"Tell me about: {title}"

            if question in seen_q_set:
                # Still duplicate — add entry ID for uniqueness
                question = f"{question} [context: {entry.get('id', '')[:6]}]"

            seen_q_set.add(question)
            answer = format_answer(content, category, thinking)
            example = build_sft_example(question, answer, category, reasoning_type)
            examples.append(example)

    print(f"  Generated {len(examples)} examples from corpus ({skipped} entries skipped)")

    # Add supplemental hand-crafted examples
    examples.extend(SUPPLEMENTAL_EXAMPLES)
    print(f"  Added {len(SUPPLEMENTAL_EXAMPLES)} hand-crafted supplemental examples")

    # Deduplicate by question text (keep first occurrence)
    seen_questions = set()
    unique_examples = []
    dup_count = 0
    for ex in examples:
        q = ex["messages"][1]["content"]
        if q not in seen_questions:
            seen_questions.add(q)
            unique_examples.append(ex)
        else:
            dup_count += 1
    examples = unique_examples
    print(f"  After dedup: {len(examples)} unique examples ({dup_count} duplicates removed)")

    # Shuffle and split train/val (90/10)
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "generated_sft_train.jsonl"
    val_path = OUTPUT_DIR / "generated_sft_val.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Statistics
    cats = {}
    reasoning_types = {}
    total_tokens = 0
    for ex in examples:
        cat = ex["category"]
        cats[cat] = cats.get(cat, 0) + 1
        rt = ex["reasoning_type"]
        reasoning_types[rt] = reasoning_types.get(rt, 0) + 1
        total_tokens += sum(len(m["content"]) for m in ex["messages"]) // 4

    print(f"\n{'='*60}")
    print(f"SFT DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")
    print(f"Total examples: {len(examples)}")
    print(f"Approx tokens:  {total_tokens:,}")
    print(f"\nBy category:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print(f"\nBy reasoning type:")
    for rt, count in sorted(reasoning_types.items(), key=lambda x: -x[1]):
        print(f"  {rt}: {count}")
    print(f"\nTrain: {train_path}")
    print(f"Val:   {val_path}")


if __name__ == "__main__":
    main()
