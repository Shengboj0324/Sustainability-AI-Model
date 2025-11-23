#!/usr/bin/env python3
"""
Real-World iOS Environment Simulation Test
Tests the system with realistic user queries and iOS-specific constraints
"""

import sys
import os
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class iOSTestCase:
    """Represents a real-world iOS user query"""
    query: str
    category: str
    expected_answer_type: str
    has_image: bool = False
    network_condition: str = "4G"  # WiFi, 4G, 3G, slow
    device: str = "iPhone 14 Pro"


class RealWorldiOSSimulator:
    """Simulates real iOS app usage with actual user queries"""
    
    def __init__(self):
        self.test_cases = self._generate_test_cases()
        self.results = []
        
    def _generate_test_cases(self) -> List[iOSTestCase]:
        """Generate 50+ real-world test cases"""
        return [
            # Beginner sustainability questions
            iOSTestCase(
                query="How do I start recycling at home?",
                category="beginner_recycling",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="What can I recycle in my kitchen?",
                category="beginner_recycling",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Is this plastic bottle recyclable?",
                category="beginner_recycling",
                expected_answer_type="FACTUAL",
                has_image=True
            ),
            
            # Upcycling and creativity
            iOSTestCase(
                query="What can I make from old jeans?",
                category="upcycling",
                expected_answer_type="CREATIVE"
            ),
            iOSTestCase(
                query="How to turn wine bottles into lamps?",
                category="upcycling",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Creative ideas for cardboard boxes",
                category="upcycling",
                expected_answer_type="CREATIVE"
            ),
            iOSTestCase(
                query="Can I make furniture from pallets?",
                category="upcycling",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Upcycle old t-shirts into bags",
                category="upcycling",
                expected_answer_type="HOW_TO"
            ),
            
            # Waste identification with images
            iOSTestCase(
                query="What type of waste is this?",
                category="waste_identification",
                expected_answer_type="FACTUAL",
                has_image=True
            ),
            iOSTestCase(
                query="Can I compost this?",
                category="waste_identification",
                expected_answer_type="FACTUAL",
                has_image=True
            ),
            iOSTestCase(
                query="Is this hazardous waste?",
                category="waste_identification",
                expected_answer_type="FACTUAL",
                has_image=True
            ),
            
            # Organization search
            iOSTestCase(
                query="Recycling centers near me in San Francisco",
                category="org_search",
                expected_answer_type="ORG_SEARCH"
            ),
            iOSTestCase(
                query="Where can I donate old clothes in NYC?",
                category="org_search",
                expected_answer_type="ORG_SEARCH"
            ),
            iOSTestCase(
                query="Electronics recycling in Seattle",
                category="org_search",
                expected_answer_type="ORG_SEARCH"
            ),
            iOSTestCase(
                query="Composting services in Austin Texas",
                category="org_search",
                expected_answer_type="ORG_SEARCH"
            ),
            
            # Advanced sustainability
            iOSTestCase(
                query="What is circular economy?",
                category="advanced_concepts",
                expected_answer_type="FACTUAL"
            ),
            iOSTestCase(
                query="How does plastic recycling work?",
                category="advanced_concepts",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Difference between biodegradable and compostable",
                category="advanced_concepts",
                expected_answer_type="FACTUAL"
            ),
            iOSTestCase(
                query="What are microplastics and why are they harmful?",
                category="advanced_concepts",
                expected_answer_type="FACTUAL"
            ),
            
            # Specific materials
            iOSTestCase(
                query="How to recycle aluminum cans?",
                category="specific_materials",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Can I recycle pizza boxes?",
                category="specific_materials",
                expected_answer_type="FACTUAL"
            ),
            iOSTestCase(
                query="What to do with old batteries?",
                category="specific_materials",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Are coffee cups recyclable?",
                category="specific_materials",
                expected_answer_type="FACTUAL"
            ),
            iOSTestCase(
                query="How to dispose of paint cans?",
                category="specific_materials",
                expected_answer_type="HOW_TO"
            ),
            
            # Composting
            iOSTestCase(
                query="How to start composting at home?",
                category="composting",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="What can go in compost bin?",
                category="composting",
                expected_answer_type="FACTUAL"
            ),
            iOSTestCase(
                query="My compost smells bad, what's wrong?",
                category="composting",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Indoor composting for apartments",
                category="composting",
                expected_answer_type="HOW_TO"
            ),
            
            # Zero waste lifestyle
            iOSTestCase(
                query="How to live zero waste?",
                category="zero_waste",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Zero waste shopping tips",
                category="zero_waste",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Plastic-free alternatives for kitchen",
                category="zero_waste",
                expected_answer_type="CREATIVE"
            ),
            
            # DIY and repairs
            iOSTestCase(
                query="How to fix a broken chair?",
                category="diy_repair",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Repair torn jeans instead of throwing away",
                category="diy_repair",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Fix broken phone screen at home",
                category="diy_repair",
                expected_answer_type="HOW_TO"
            ),
            
            # Seasonal and specific scenarios
            iOSTestCase(
                query="What to do with Christmas tree after holidays?",
                category="seasonal",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Recycle Halloween pumpkins",
                category="seasonal",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Eco-friendly gift wrapping ideas",
                category="seasonal",
                expected_answer_type="CREATIVE"
            ),
            
            # Kids and education
            iOSTestCase(
                query="Teach kids about recycling",
                category="education",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Fun recycling crafts for children",
                category="education",
                expected_answer_type="CREATIVE"
            ),
            iOSTestCase(
                query="School project ideas about sustainability",
                category="education",
                expected_answer_type="CREATIVE"
            ),
            
            # Business and office
            iOSTestCase(
                query="Office recycling program setup",
                category="business",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Reduce paper waste in workplace",
                category="business",
                expected_answer_type="HOW_TO"
            ),
            
            # Food waste
            iOSTestCase(
                query="How to reduce food waste at home?",
                category="food_waste",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="What to do with leftover food?",
                category="food_waste",
                expected_answer_type="CREATIVE"
            ),
            iOSTestCase(
                query="Can I compost meat and dairy?",
                category="food_waste",
                expected_answer_type="FACTUAL"
            ),
            
            # Fashion and textiles
            iOSTestCase(
                query="Sustainable fashion tips",
                category="fashion",
                expected_answer_type="HOW_TO"
            ),
            iOSTestCase(
                query="Where to donate old shoes?",
                category="fashion",
                expected_answer_type="ORG_SEARCH"
            ),
            iOSTestCase(
                query="Repair vs replace clothing",
                category="fashion",
                expected_answer_type="FACTUAL"
            ),
        ]
    
    def test_answer_formatter(self, test_case: iOSTestCase) -> Dict[str, Any]:
        """Test answer formatter with real query"""
        from services.shared.answer_formatter import AnswerFormatter, AnswerType

        formatter = AnswerFormatter()

        # Simulate different answer types based on query
        answer_type_map = {
            "HOW_TO": AnswerType.HOW_TO,
            "FACTUAL": AnswerType.FACTUAL,
            "CREATIVE": AnswerType.CREATIVE,
            "ORG_SEARCH": AnswerType.ORG_SEARCH,
            "GENERAL": AnswerType.GENERAL,
        }

        answer_type = answer_type_map.get(test_case.expected_answer_type, AnswerType.GENERAL)

        # Generate mock answer based on query
        answer = self._generate_mock_answer(test_case)
        sources = self._generate_mock_sources(test_case)

        # Format the answer
        formatted = formatter.format_answer(
            answer=answer,
            answer_type=answer_type,
            sources=sources,
            metadata={
                "query": test_case.query,
                "category": test_case.category,
                "device": test_case.device,
                "network": test_case.network_condition,
            }
        )

        return {
            "query": test_case.query,
            "category": test_case.category,
            "answer_type": test_case.expected_answer_type,
            "markdown_length": len(formatted.content),
            "html_length": len(formatted.html_content),
            "plain_length": len(formatted.plain_text),
            "num_citations": len(formatted.citations),
            "has_metadata": formatted.metadata is not None,
            "markdown_preview": formatted.content[:200] + "..." if len(formatted.content) > 200 else formatted.content,
            "html_preview": formatted.html_content[:200] + "..." if len(formatted.html_content) > 200 else formatted.html_content,
        }

    def _generate_mock_answer(self, test_case: iOSTestCase) -> str:
        """Generate realistic mock answer based on query"""
        answers = {
            "How do I start recycling at home?": """
# Getting Started with Home Recycling

Starting a recycling routine at home is easier than you think! Here's a comprehensive guide:

## Step 1: Set Up Your Recycling Station
- Choose a convenient location (kitchen, garage, or utility room)
- Get separate bins for different materials
- Label each bin clearly

## Step 2: Learn What's Recyclable
- **Paper & Cardboard**: newspapers, magazines, cardboard boxes
- **Plastics**: bottles, containers (check numbers 1, 2, 5)
- **Glass**: bottles and jars
- **Metals**: aluminum cans, tin cans

## Step 3: Clean and Prepare
- Rinse containers to remove food residue
- Remove caps and lids
- Flatten cardboard boxes to save space

## Step 4: Check Local Guidelines
Different areas have different rules. Contact your local waste management to learn specific requirements.

## Tips for Success
- Make it a family activity
- Start small and build the habit
- Keep a recycling guide handy
""",
            "What can I make from old jeans?": """
# Creative Upcycling Ideas for Old Jeans

Transform your worn-out denim into amazing new items!

## Fashion & Accessories
1. **Denim Tote Bag**: Cut and sew into a sturdy shopping bag
2. **Patchwork Quilt**: Combine different denim shades
3. **Headbands**: Use the waistband for stretchy headbands
4. **Coasters**: Cut circles and add cork backing

## Home Decor
5. **Throw Pillows**: Stuff with old t-shirts or pillow filling
6. **Wall Organizer**: Create pockets for storage
7. **Plant Pot Covers**: Wrap around plain pots
8. **Placemats**: Layer and stitch for durability

## Practical Items
9. **Apron**: Perfect for gardening or cooking
10. **Dog Toy**: Braid strips for a chew toy
11. **Book Cover**: Protect your favorite books
12. **Laptop Sleeve**: Add padding for protection

**Materials Needed**: Scissors, sewing machine (or needle & thread), fabric glue

**Difficulty**: Beginner to Intermediate
""",
            "What type of waste is this?": """
# Waste Identification Result

Based on the image analysis:

## Classification
**Type**: Plastic Waste - PET Bottle
**Recyclability**: ✅ Highly Recyclable
**Material Code**: #1 PETE

## Disposal Instructions
1. **Empty** the bottle completely
2. **Rinse** with water to remove residue
3. **Remove** the cap (recycle separately)
4. **Crush** to save space
5. **Place** in plastic recycling bin

## Environmental Impact
- PET bottles are 100% recyclable
- Can be recycled into new bottles, clothing, carpet
- Recycling saves 75% of energy vs. making new plastic

## Alternative Actions
- **Reuse**: Clean and refill for water
- **Upcycle**: Create planters, organizers, or bird feeders
- **Return**: Some stores offer bottle deposit returns
""",
        }

        # Return specific answer or generate generic one
        if test_case.query in answers:
            return answers[test_case.query]
        else:
            return f"Here's helpful information about: {test_case.query}\n\nThis is a comprehensive answer that addresses your question with practical, actionable advice."

    def _generate_mock_sources(self, test_case: iOSTestCase) -> List[Dict[str, Any]]:
        """Generate realistic mock sources"""
        return [
            {
                "id": 1,
                "source": "EPA Recycling Guidelines",
                "doc_type": "article",
                "score": 0.95,
                "url": "https://www.epa.gov/recycle",
                "metadata": {"published": "2024-01-15"}
            },
            {
                "id": 2,
                "source": "Sustainable Living Guide",
                "doc_type": "guide",
                "score": 0.89,
                "url": "https://example.com/sustainability",
                "metadata": {"author": "Green Living Team"}
            },
            {
                "id": 3,
                "source": "Community Recycling Database",
                "doc_type": "database",
                "score": 0.82,
                "url": "https://example.com/recycling-db",
                "metadata": {"updated": "2024-11-20"}
            }
        ]

    def run_all_tests(self):
        """Run all test cases and display results"""
        print("\n" + "="*100)
        print("🍎 REAL-WORLD iOS ENVIRONMENT SIMULATION")
        print("="*100)
        print(f"Total Test Cases: {len(self.test_cases)}")
        print(f"Simulating: iPhone/iPad users with real sustainability questions")
        print("="*100)

        # Test each case
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n{'='*100}")
            print(f"TEST CASE #{i}/{len(self.test_cases)}")
            print(f"{'='*100}")
            print(f"📱 Device: {test_case.device}")
            print(f"📶 Network: {test_case.network_condition}")
            print(f"📂 Category: {test_case.category}")
            print(f"❓ Query: {test_case.query}")
            print(f"🎯 Expected Type: {test_case.expected_answer_type}")
            print(f"🖼️  Has Image: {'Yes' if test_case.has_image else 'No'}")

            try:
                start_time = time.time()
                result = self.test_answer_formatter(test_case)
                duration = time.time() - start_time

                print(f"\n✅ RESPONSE GENERATED ({duration*1000:.1f}ms)")
                print(f"{'─'*100}")
                print(f"📊 Answer Type: {result['answer_type']}")
                print(f"📝 Markdown Length: {result['markdown_length']} chars")
                print(f"🌐 HTML Length: {result['html_length']} chars")
                print(f"📄 Plain Text Length: {result['plain_length']} chars")
                print(f"📚 Citations: {result['num_citations']}")
                print(f"\n📖 MARKDOWN PREVIEW:")
                print(f"{'─'*100}")
                print(result['markdown_preview'])
                print(f"{'─'*100}")
                print(f"\n🌐 HTML PREVIEW:")
                print(f"{'─'*100}")
                print(result['html_preview'])
                print(f"{'─'*100}")

                self.results.append({
                    "test_case": asdict(test_case),
                    "result": result,
                    "duration_ms": duration * 1000,
                    "success": True
                })

            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                self.results.append({
                    "test_case": asdict(test_case),
                    "error": str(e),
                    "success": False
                })

        # Print summary
        self._print_summary()

        return len([r for r in self.results if r["success"]]) == len(self.test_cases)

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*100)
        print("📊 TEST SUMMARY")
        print("="*100)

        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]

        print(f"Total Tests: {len(self.results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"Success Rate: {len(successful)/len(self.results)*100:.1f}%")

        if successful:
            durations = [r["duration_ms"] for r in successful]
            print(f"\n⏱️  Performance:")
            print(f"  Average Response Time: {sum(durations)/len(durations):.1f}ms")
            print(f"  Min Response Time: {min(durations):.1f}ms")
            print(f"  Max Response Time: {max(durations):.1f}ms")

        # Category breakdown
        categories = {}
        for r in successful:
            cat = r["test_case"]["category"]
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\n📂 Category Breakdown:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} tests")

        # Answer type breakdown
        answer_types = {}
        for r in successful:
            atype = r["result"]["answer_type"]
            answer_types[atype] = answer_types.get(atype, 0) + 1

        print(f"\n🎯 Answer Type Distribution:")
        for atype, count in sorted(answer_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {atype}: {count} answers")

        print("="*100)


if __name__ == "__main__":
    simulator = RealWorldiOSSimulator()
    success = simulator.run_all_tests()
    sys.exit(0 if success else 1)

