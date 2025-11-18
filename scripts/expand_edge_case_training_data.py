"""
Expand Training Data with Edge Cases and Rare Scenarios

CRITICAL: Prepares the model to handle ANY possible user input including:
- Ultra-rare edge cases
- Ambiguous questions
- Multi-step reasoning
- Error recovery scenarios
- Low-quality or incomplete information
- Multi-modal combinations
- Unusual materials and items
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def create_edge_case_examples() -> List[Dict[str, Any]]:
    """
    Create comprehensive edge case training examples
    
    Returns: List of training examples in instruction format
    """
    examples = []
    
    # 1. AMBIGUOUS QUESTIONS
    examples.extend([
        {
            "instruction": "What do I do with this?",
            "input": "",
            "output": "I'd be happy to help! To give you the best advice, could you tell me more about what 'this' is? For example:\n- What material is it made of?\n- What type of item is it?\n- Or you could upload a photo!\n\nOnce I know more, I can tell you whether it should be recycled, composted, thrown away, or even upcycled into something new!",
            "category": "ambiguous_handling"
        },
        {
            "instruction": "Can I recycle it?",
            "input": "",
            "output": "I'd love to help you figure that out! To determine if something can be recycled, I need to know:\n- What material it's made from (plastic, metal, glass, paper, etc.)\n- What type of item it is\n- Your location (recycling rules vary by area)\n\nCould you provide more details or upload a photo? That way I can give you accurate recycling guidance!",
            "category": "ambiguous_handling"
        },
        {
            "instruction": "Is this safe?",
            "input": "",
            "output": "Safety is important! To assess whether something is safe, I need more information:\n- What item are you asking about?\n- Safe for what purpose? (handling, disposal, upcycling, etc.)\n- What material is it made from?\n\nPlease provide more details or a photo, and I'll help you understand any safety concerns!",
            "category": "ambiguous_handling"
        }
    ])
    
    # 2. ULTRA-RARE MATERIALS
    examples.extend([
        {
            "instruction": "How do I dispose of asbestos insulation?",
            "input": "",
            "output": "⚠️ CRITICAL SAFETY WARNING: Asbestos is a HAZARDOUS material that requires professional handling!\n\nDO NOT attempt to remove or dispose of asbestos yourself. Asbestos fibers can cause serious lung diseases including cancer.\n\nWhat you MUST do:\n1. Contact a licensed asbestos abatement professional\n2. Do not disturb the material\n3. Keep the area sealed off\n4. Contact your local environmental health department\n5. Never put asbestos in regular trash\n\nThis is a regulated hazardous waste that requires special permits and disposal procedures. Your safety is paramount!",
            "category": "hazardous_rare"
        },
        {
            "instruction": "What bin for mercury thermometer?",
            "input": "",
            "output": "⚠️ Mercury thermometers contain TOXIC mercury and require special handling!\n\nDO NOT put in any regular bin. Mercury is hazardous waste.\n\nProper disposal:\n1. Place in a sealed plastic bag or container\n2. Take to a household hazardous waste facility\n3. Many pharmacies accept old thermometers\n4. Contact your local waste management for collection events\n5. NEVER throw in trash or recycling\n\nIf broken: Ventilate the area, avoid touching mercury, and contact your local poison control center immediately.",
            "category": "hazardous_rare"
        },
        {
            "instruction": "Can I recycle carbon fiber bike frame?",
            "input": "",
            "output": "Carbon fiber is a challenging material to recycle! Here's what you need to know:\n\nRecycling Reality:\n- Most municipal recycling programs DON'T accept carbon fiber\n- Carbon fiber recycling requires specialized facilities\n- Very few locations can process it\n\nBetter Options:\n1. **Repair**: Many bike shops can repair carbon fiber frames\n2. **Resell**: Sell or donate if still functional\n3. **Upcycle**: Use as art, furniture, or decorative pieces\n4. **Specialized Recyclers**: Contact manufacturers like Trek or Specialized - some have take-back programs\n5. **Last Resort**: Landfill (unfortunately)\n\nCarbon fiber recycling technology is improving, so check with local specialty recyclers or the manufacturer!",
            "category": "rare_materials"
        }
    ])
    
    # 3. MULTI-STEP REASONING
    examples.extend([
        {
            "instruction": "I have a broken laptop. The screen is cracked, battery doesn't work, but the hard drive might be okay. What should I do?",
            "input": "",
            "output": "Great question! Let's break this down step-by-step:\n\n**Step 1: Data Security (CRITICAL)**\n- Remove the hard drive first\n- Wipe data or keep for data recovery\n- Never recycle electronics with personal data!\n\n**Step 2: Battery Safety**\n- Remove the battery if possible (lithium batteries are fire hazards)\n- Take battery to battery recycling (Best Buy, Home Depot, etc.)\n- NEVER put batteries in regular trash\n\n**Step 3: Laptop Disposal Options**\nA) **E-Waste Recycling** (BEST):\n   - Take to certified e-waste recycler\n   - Best Buy, Staples offer free recycling\n   - Manufacturer take-back programs\n\nB) **Donation for Parts**:\n   - Local repair shops may want it for parts\n   - Schools/nonprofits for educational teardowns\n\nC) **Upcycling Ideas**:\n   - Use keyboard as external keyboard\n   - Screen frame as picture frame\n   - Internal components for art projects\n\n**Step 4: What NOT to do**:\n- ❌ Don't put in regular trash\n- ❌ Don't put in curbside recycling\n- ❌ Don't leave battery inside\n\nPriority: Data security → Battery safety → Proper e-waste recycling!",
            "category": "multi_step_reasoning"
        },
        {
            "instruction": "I want to upcycle wine bottles but some have labels that won't come off and some are different colors. What's the best approach?",
            "input": "",
            "output": "Excellent upcycling project! Let's tackle this systematically:\n\n**Step 1: Label Removal**\nFor stubborn labels, try:\n- Soak in hot soapy water for 30 minutes\n- Use baking soda paste + scrubbing\n- Acetone (nail polish remover) for adhesive residue\n- Commercial adhesive removers (Goo Gone)\n- OR embrace the labels as part of the design!\n\n**Step 2: Color Sorting**\nDifferent colors = different opportunities:\n- **Clear bottles**: Best for vases, lamps, terrariums\n- **Green bottles**: Garden borders, candle holders\n- **Brown bottles**: Rustic decor, tiki torches\n- **Mixed colors**: Mosaic projects, wind chimes\n\n**Step 3: Project Ideas by Color**\n\n*Clear Bottles*:\n- Elegant vases\n- Oil/vinegar dispensers\n- Drinking glasses (cut and sand)\n- Pendant lights\n\n*Colored Bottles*:\n- Garden edging\n- Bottle trees\n- Stained glass effect windows\n- Tiki torches (outdoor)\n\n**Step 4: Safety Tips**\n- Wear gloves when cutting glass\n- Sand all edges smooth\n- Use proper glass-cutting tools\n- Work in ventilated area\n\n**Pro Tip**: Group by color for cohesive projects, or mix intentionally for eclectic designs!",
            "category": "multi_step_reasoning"
        }
    ])

    # 4. LOW-QUALITY / INCOMPLETE INFORMATION
    examples.extend([
        {
            "instruction": "plastic thing",
            "input": "",
            "output": "I can help with that plastic item! To give you the best guidance, I need a bit more information:\n\n**What I need to know:**\n- What type of plastic item? (bottle, container, bag, toy, etc.)\n- Is there a recycling number? (look for ♻️ with a number 1-7)\n- What size is it?\n- What condition is it in?\n\n**Or even better**: Upload a photo and I can identify it!\n\n**General plastic recycling tips:**\n- #1 (PET) and #2 (HDPE): Usually recyclable\n- #3-7: Check local guidelines\n- Clean and dry before recycling\n- Remove caps and labels if possible\n\nLet me know more details and I'll give you specific advice!",
            "category": "incomplete_info"
        },
        {
            "instruction": "old stuff from garage",
            "input": "",
            "output": "Garage cleanouts can have all sorts of items! Let's figure out what to do with your stuff:\n\n**Common garage items and disposal:**\n\n**Hazardous (Special Handling)**:\n- Paint, solvents, chemicals → Hazardous waste facility\n- Motor oil, antifreeze → Auto parts stores\n- Batteries → Battery recycling\n- Pesticides → Hazardous waste collection\n\n**Recyclable**:\n- Metal tools, parts → Scrap metal recycling\n- Cardboard boxes → Curbside recycling\n- Plastic containers → Check recycling number\n\n**Reusable/Donate**:\n- Working tools → Habitat for Humanity ReStore\n- Sports equipment → Goodwill, Play It Again Sports\n- Furniture → Donation centers\n\n**Could you tell me:**\n- What specific items do you have?\n- Are any of them hazardous (chemicals, paint, etc.)?\n- Are any still in working condition?\n\nOr send photos and I'll help you sort through everything!",
            "category": "incomplete_info"
        }
    ])

    # 5. UNUSUAL COMBINATIONS
    examples.extend([
        {
            "instruction": "I have a leather jacket with metal zippers and plastic buttons. How do I recycle this?",
            "input": "",
            "output": "Great question! Multi-material items like this are tricky for recycling. Here's your best approach:\n\n**Reality Check:**\n- Mixed-material items are NOT accepted in regular recycling\n- Separation is usually not worth the effort\n- Textile recycling programs rarely accept leather\n\n**BETTER OPTIONS (in order of preference):**\n\n**1. Donate/Resell** (BEST if wearable):\n- Thrift stores (Goodwill, Salvation Army)\n- Consignment shops for leather\n- Online: Poshmark, ThredUp, eBay\n- Leather is valuable - someone will want it!\n\n**2. Repair**:\n- Leather repair shops can fix zippers\n- Replace buttons easily\n- Leather lasts decades if maintained\n\n**3. Upcycle**:\n- Cut into smaller leather pieces for crafts\n- Remove zippers for future projects\n- Make patches, wallets, or accessories\n- Leather is premium upcycling material!\n\n**4. Textile Recycling**:\n- H&M, North Face accept textiles\n- Some programs take leather (call first)\n- Remove metal parts if possible\n\n**5. Last Resort - Landfill**:\n- If unwearable and can't donate\n- Leather is biodegradable (eventually)\n\n**Bottom line**: Leather jackets have value - donate or upcycle before considering disposal!",
            "category": "multi_material"
        },
        {
            "instruction": "Broken phone with cracked screen and water damage. Battery swollen. What do I do?",
            "input": "",
            "output": "⚠️ **SAFETY FIRST**: Swollen battery is a FIRE HAZARD! Handle carefully.\n\n**IMMEDIATE ACTIONS:**\n\n**1. Battery Safety (URGENT)**:\n- Do NOT charge the phone\n- Do NOT press on swollen battery\n- Keep away from heat\n- Store in cool, dry place\n- If battery is bulging significantly, place in fireproof container\n\n**2. Data Security**:\n- If phone still works, backup data NOW\n- If not working, data may be lost\n- Professional data recovery is expensive ($300-$1000+)\n\n**3. Disposal Options**:\n\n**A) Manufacturer Take-Back** (BEST):\n- Apple: Free recycling, may offer credit\n- Samsung: Trade-in or recycling program\n- Google: Recycling program\n\n**B) Retailer Recycling**:\n- Best Buy: Free e-waste recycling\n- Staples: Free electronics recycling\n- Carrier stores (AT&T, Verizon, T-Mobile)\n\n**C) Specialized E-Waste**:\n- Call2Recycle for battery\n- Local e-waste recycling events\n- Certified e-waste recyclers\n\n**WHAT NOT TO DO**:\n- ❌ Don't put in regular trash (illegal in many states)\n- ❌ Don't put in curbside recycling\n- ❌ Don't try to remove battery yourself (fire risk)\n- ❌ Don't mail it (swollen batteries can't be shipped)\n\n**Priority**: Battery safety → Data backup → Proper e-waste recycling\n\nSwollen batteries can catch fire - handle this TODAY!",
            "category": "complex_hazardous"
        }
    ])

    # 6. ERROR RECOVERY / CORRECTIONS
    examples.extend([
        {
            "instruction": "Wait, I said it was plastic but it's actually metal. Does that change anything?",
            "input": "",
            "output": "Yes, that definitely changes things! Metal is actually EASIER to recycle than plastic. Here's the good news:\n\n**Metal Recycling (Much Better!):**\n\n**Advantages**:\n- Metal can be recycled infinitely without quality loss\n- Higher recycling rates than plastic\n- More valuable - recyclers want it!\n- Accepted almost everywhere\n\n**What to do**:\n1. **Curbside Recycling**: Most metals accepted (aluminum, steel, tin)\n2. **Scrap Metal Recycling**: For larger items, you might even get paid!\n3. **Magnet Test**: \n   - Sticks to magnet = Steel/iron (recyclable)\n   - Doesn't stick = Aluminum, copper, brass (recyclable, more valuable)\n\n**Preparation**:\n- Rinse if it held food\n- Remove any plastic/rubber parts if easy\n- Flatten cans to save space\n\n**Special Cases**:\n- **Aluminum cans**: Definitely recycle! High value\n- **Steel cans**: Recycle (soup, vegetable cans)\n- **Scrap metal**: Take to scrap yard for cash\n- **Mixed metal items**: Usually still recyclable\n\nMetal is one of the BEST things to recycle - definitely don't throw it away!",
            "category": "error_correction"
        },
        {
            "instruction": "Sorry, I meant to ask about composting, not recycling",
            "input": "",
            "output": "No problem! Composting is completely different from recycling. Let me help with composting instead:\n\n**Composting Basics:**\n\n**CAN Compost (Greens & Browns)**:\n- Fruit and vegetable scraps\n- Coffee grounds and filters\n- Tea bags (remove staples)\n- Eggshells\n- Yard waste (leaves, grass)\n- Shredded paper/cardboard\n- Plant trimmings\n\n**CANNOT Compost**:\n- Meat, fish, bones\n- Dairy products\n- Oils and fats\n- Pet waste\n- Diseased plants\n- Glossy/coated paper\n\n**Composting Options**:\n\n**1. Backyard Composting**:\n- Build or buy a compost bin\n- Mix greens (nitrogen) and browns (carbon)\n- Keep moist, turn regularly\n- Ready in 2-6 months\n\n**2. Municipal Composting**:\n- Many cities offer curbside compost pickup\n- Accepts more items than home composting\n- Check local program rules\n\n**3. Vermicomposting** (worms):\n- Great for apartments\n- Worms eat food scraps\n- Produces rich fertilizer\n\n**What specifically do you want to compost?** I can give you detailed guidance!",
            "category": "error_correction"
        }
    ])

    # 7. MULTI-LANGUAGE HINTS (English examples with common non-English terms)
    examples.extend([
        {
            "instruction": "¿Cómo reciclo botellas de plástico? (How do I recycle plastic bottles?)",
            "input": "",
            "output": "¡Great question! I can help you recycle plastic bottles:\n\n**Plastic Bottle Recycling:**\n\n**Step 1: Check the Number**\n- Look for ♻️ symbol with number\n- #1 (PET/PETE): Water, soda bottles - HIGHLY recyclable\n- #2 (HDPE): Milk, detergent bottles - HIGHLY recyclable\n- Other numbers: Check local guidelines\n\n**Step 2: Prepare for Recycling**\n1. Empty completely\n2. Rinse with water\n3. Remove caps (recycle separately if #2 or #5)\n4. Remove labels (optional, but helpful)\n5. Crush to save space\n\n**Step 3: Recycle**\n- Put in curbside recycling bin\n- Or take to recycling center\n- Some states have bottle deposits (CA, NY, etc.) - return for money!\n\n**Better Options**:\n- **Reuse**: Refill water bottles\n- **Upcycle**: Planters, organizers, bird feeders\n- **Reduce**: Use reusable bottles instead\n\n**¿Necesitas más ayuda?** (Need more help?) Let me know!",
            "category": "multi_language"
        }
    ])

    return examples


def main():
    """Generate and save edge case training data"""
    print("=" * 80)
    print("GENERATING EDGE CASE TRAINING DATA")
    print("=" * 80)
    
    # Create examples
    edge_cases = create_edge_case_examples()
    
    print(f"\n✅ Generated {len(edge_cases)} edge case examples")
    print(f"   - Ambiguous handling: {sum(1 for e in edge_cases if e['category'] == 'ambiguous_handling')}")
    print(f"   - Hazardous/rare materials: {sum(1 for e in edge_cases if 'hazardous' in e['category'] or 'rare' in e['category'])}")
    print(f"   - Multi-step reasoning: {sum(1 for e in edge_cases if e['category'] == 'multi_step_reasoning')}")
    
    # Load existing data
    existing_file = PROJECT_ROOT / "data" / "llm_training_expanded.json"
    with open(existing_file, 'r') as f:
        existing_data = json.load(f)
    
    print(f"\n📊 Existing training data: {len(existing_data)} examples")
    
    # Combine
    combined_data = existing_data + edge_cases
    
    # Save combined
    output_file = PROJECT_ROOT / "data" / "llm_training_ultra_expanded.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"\n✅ Saved {len(combined_data)} total examples to: {output_file}")
    print(f"   - Original: {len(existing_data)}")
    print(f"   - New edge cases: {len(edge_cases)}")
    print(f"   - Total: {len(combined_data)}")
    
    print("\n" + "=" * 80)
    print("✅ EDGE CASE DATA GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

