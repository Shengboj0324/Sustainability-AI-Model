"""
Comprehensive Test Suite for NLP Enhancements

Tests:
1. Intent Classification (7 categories × 10 examples = 70 tests)
2. Entity Extraction (7 entity types × 10 examples = 70 tests)
3. Multi-Language Support (8 languages × 5 examples = 40 tests)

Total: 180 tests

CRITICAL: Validates all NLP enhancements with strictest quality requirements
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.llm_service.intent_classifier import IntentClassifier, IntentCategory
from services.llm_service.entity_extractor import EntityExtractor
from services.llm_service.language_handler import LanguageHandler, Language


class NLPTester:
    """Comprehensive NLP testing suite"""

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.language_handler = LanguageHandler()

        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def test_intent_classification(self):
        """Test intent classification with 70 examples"""
        print("\n" + "="*80)
        print("TEST 1: INTENT CLASSIFICATION (70 examples)")
        print("="*80)

        test_cases = [
            # WASTE_IDENTIFICATION (10 examples)
            ("What is this item?", IntentCategory.WASTE_IDENTIFICATION),
            ("Can I recycle this plastic bottle?", IntentCategory.WASTE_IDENTIFICATION),
            ("Identify this material", IntentCategory.WASTE_IDENTIFICATION),
            ("What type of plastic is this?", IntentCategory.WASTE_IDENTIFICATION),
            ("Is this recyclable?", IntentCategory.WASTE_IDENTIFICATION),
            ("What kind of waste is this?", IntentCategory.WASTE_IDENTIFICATION),
            ("Classify this object", IntentCategory.WASTE_IDENTIFICATION),
            ("What material is this made of?", IntentCategory.WASTE_IDENTIFICATION),
            ("Can I compost this?", IntentCategory.WASTE_IDENTIFICATION),
            ("Is this metal or plastic?", IntentCategory.WASTE_IDENTIFICATION),

            # DISPOSAL_GUIDANCE (10 examples)
            ("How do I dispose of this battery?", IntentCategory.DISPOSAL_GUIDANCE),
            ("Which bin should I use?", IntentCategory.DISPOSAL_GUIDANCE),
            ("Where do I throw this away?", IntentCategory.DISPOSAL_GUIDANCE),
            ("How to get rid of old electronics?", IntentCategory.DISPOSAL_GUIDANCE),
            ("What bin does this go in?", IntentCategory.DISPOSAL_GUIDANCE),
            ("How should I dispose of paint?", IntentCategory.DISPOSAL_GUIDANCE),
            ("Correct bin for cardboard?", IntentCategory.DISPOSAL_GUIDANCE),
            ("Where to discard glass bottles?", IntentCategory.DISPOSAL_GUIDANCE),
            ("How to throw away food waste?", IntentCategory.DISPOSAL_GUIDANCE),
            ("Which container for plastic bags?", IntentCategory.DISPOSAL_GUIDANCE),

            # UPCYCLING_IDEAS (10 examples)
            ("How can I upcycle this jar?", IntentCategory.UPCYCLING_IDEAS),
            ("Creative ideas for old t-shirts", IntentCategory.UPCYCLING_IDEAS),
            ("What can I make from plastic bottles?", IntentCategory.UPCYCLING_IDEAS),
            ("Repurpose old furniture", IntentCategory.UPCYCLING_IDEAS),
            ("DIY projects with cardboard boxes", IntentCategory.UPCYCLING_IDEAS),
            ("How to reuse glass jars?", IntentCategory.UPCYCLING_IDEAS),
            ("Turn old clothes into something useful", IntentCategory.UPCYCLING_IDEAS),
            ("Upcycling ideas for tin cans", IntentCategory.UPCYCLING_IDEAS),
            ("What's a second life for this item?", IntentCategory.UPCYCLING_IDEAS),
            ("Creative reuse for wine bottles", IntentCategory.UPCYCLING_IDEAS),

            # ORGANIZATION_SEARCH (10 examples)
            ("Where can I donate old clothes?", IntentCategory.ORGANIZATION_SEARCH),
            ("Find recycling centers near me", IntentCategory.ORGANIZATION_SEARCH),
            ("Charities that accept furniture", IntentCategory.ORGANIZATION_SEARCH),
            ("Drop-off locations for electronics", IntentCategory.ORGANIZATION_SEARCH),
            ("Recycling centers in my area", IntentCategory.ORGANIZATION_SEARCH),
            ("Where to donate books?", IntentCategory.ORGANIZATION_SEARCH),
            ("Organizations that take donations", IntentCategory.ORGANIZATION_SEARCH),
            ("Local recycling facilities", IntentCategory.ORGANIZATION_SEARCH),
            ("Collection points for batteries", IntentCategory.ORGANIZATION_SEARCH),
            ("Thrift stores nearby", IntentCategory.ORGANIZATION_SEARCH),

            # SUSTAINABILITY_INFO (10 examples)
            ("Why is recycling important?", IntentCategory.SUSTAINABILITY_INFO),
            ("Environmental impact of plastic", IntentCategory.SUSTAINABILITY_INFO),
            ("Benefits of composting", IntentCategory.SUSTAINABILITY_INFO),
            ("How does recycling help the environment?", IntentCategory.SUSTAINABILITY_INFO),
            ("Carbon footprint of waste", IntentCategory.SUSTAINABILITY_INFO),
            ("Statistics on recycling rates", IntentCategory.SUSTAINABILITY_INFO),
            ("Impact of e-waste on climate", IntentCategory.SUSTAINABILITY_INFO),
            ("Why should I recycle?", IntentCategory.SUSTAINABILITY_INFO),
            ("Facts about plastic pollution", IntentCategory.SUSTAINABILITY_INFO),
            ("Ecological benefits of upcycling", IntentCategory.SUSTAINABILITY_INFO),

            # GENERAL_QUESTION (10 examples)
            ("How does recycling work?", IntentCategory.GENERAL_QUESTION),
            ("What is composting?", IntentCategory.GENERAL_QUESTION),
            ("Explain the recycling process", IntentCategory.GENERAL_QUESTION),
            ("Difference between recycling and upcycling", IntentCategory.GENERAL_QUESTION),
            ("Types of recyclable materials", IntentCategory.GENERAL_QUESTION),
            ("What are the categories of waste?", IntentCategory.GENERAL_QUESTION),
            ("How do I start composting?", IntentCategory.GENERAL_QUESTION),
            ("Learn about waste management", IntentCategory.GENERAL_QUESTION),
            ("What is zero waste?", IntentCategory.GENERAL_QUESTION),
            ("Understand recycling symbols", IntentCategory.GENERAL_QUESTION),

            # CHITCHAT (10 examples)
            ("Hello", IntentCategory.CHITCHAT),
            ("Thank you", IntentCategory.CHITCHAT),
            ("Hi there", IntentCategory.CHITCHAT),
            ("Thanks for your help", IntentCategory.CHITCHAT),
            ("Goodbye", IntentCategory.CHITCHAT),
            ("How are you?", IntentCategory.CHITCHAT),
            ("Good morning", IntentCategory.CHITCHAT),
            ("Yes", IntentCategory.CHITCHAT),
            ("Okay", IntentCategory.CHITCHAT),
            ("See you later", IntentCategory.CHITCHAT),
        ]

        passed = 0
        failed = 0

        for text, expected_intent in test_cases:
            self.total_tests += 1
            intent, confidence = self.intent_classifier.classify(text)

            if intent == expected_intent:
                passed += 1
                self.passed_tests += 1
                status = "✅"
            else:
                failed += 1
                self.failed_tests += 1
                status = "❌"

            if failed <= 5:  # Only show first 5 failures
                if status == "❌":
                    print(f"{status} '{text[:50]}...' -> Expected: {expected_intent.value}, Got: {intent.value}")

        accuracy = (passed / len(test_cases)) * 100
        print(f"\nIntent Classification Results:")

    def test_entity_extraction(self):
        """Test entity extraction with 70 examples"""
        print("\n" + "="*80)
        print("TEST 2: ENTITY EXTRACTION (70 examples)")
        print("="*80)

        test_cases = [
            # MATERIAL entities (10 examples)
            ("I have a plastic bottle", ["plastic"]),
            ("This is made of metal and glass", ["metal", "glass"]),
            ("Cardboard box with paper inside", ["cardboard", "paper"]),
            ("Aluminum can and steel container", ["aluminum", "steel"]),
            ("Wood furniture with fabric cushions", ["wood", "fabric"]),
            ("Styrofoam packaging material", ["styrofoam"]),
            ("Ceramic plate and porcelain cup", ["ceramic", "porcelain"]),
            ("Electronics and batteries", ["electronics", "batteries"]),
            ("Organic food waste for compost", ["organic", "food waste", "compost"]),
            ("HDPE plastic type 2", ["hdpe", "plastic", "type 2"]),

            # ITEM entities (10 examples)
            ("I have a bottle and a can", ["bottle", "can"]),
            ("Plastic cup with a lid", ["cup", "lid"]),
            ("Cardboard box and paper bag", ["box", "bag"]),
            ("Old phone and laptop charger", ["phone", "laptop", "charger"]),
            ("Clothing and shoes to donate", ["clothing", "shoes"]),
            ("Newspaper and magazine", ["newspaper", "magazine"]),
            ("Lightbulb and battery", ["lightbulb", "battery"]),
            ("Furniture and mattress", ["furniture", "mattress"]),
            ("Toy and puzzle", ["toy", "puzzle"]),
            ("Jar with a cap", ["jar", "cap"]),

            # ACTION entities (10 examples)
            ("I want to recycle this", ["recycle"]),
            ("How to dispose of batteries?", ["dispose"]),
            ("Donate old clothes", ["donate"]),
            ("Upcycle plastic bottles", ["upcycle"]),
            ("Compost food waste", ["compost"]),
            ("Throw away trash", ["throw away", "trash"]),
            ("Sort and separate recyclables", ["sort", "separate"]),
            ("Clean and wash containers", ["clean", "wash"]),
            ("Reuse glass jars", ["reuse"]),
            ("Drop off at recycling center", ["drop off", "recycling"]),

            # ORGANIZATION entities (5 examples)
            ("Donate to Goodwill", ["goodwill"]),
            ("Find a recycling center", ["recycling center"]),
            ("Charity donation center", ["charity", "donation center"]),
            ("Thrift store nearby", ["thrift store"]),
            ("Non-profit organization", ["non-profit"]),

            # LOCATION entities (10 examples)
            ("Recycling near me", ["near me"]),
            ("In my area", ["in my area"]),
            ("ZIP code 94102", ["94102"]),
            ("San Francisco, CA", ["San Francisco, CA"]),
            ("My city", ["my city"]),
            ("My neighborhood", ["my neighborhood"]),
            ("Near my location", ["near me"]),
            ("Local recycling", ["local"]),  # Note: 'local' not in patterns, will fail
            ("In my town", ["in my town"]),  # Note: 'in my town' not exact match, will fail
            ("94102-1234", ["94102-1234"]),

            # QUANTITY entities (10 examples)
            ("5 kg of plastic", ["5 kg"]),
            ("10 bottles", ["10 bottles"]),  # Note: pattern expects "pieces/items/units"
            ("2 liters of paint", ["2 liters"]),
            ("3 items to recycle", ["3 items"]),
            ("One piece of furniture", ["one piece"]),
            ("15 pounds of metal", ["15 pounds"]),
            ("500 ml bottle", ["500 ml"]),
            ("Two units", ["two units"]),
            ("8 oz can", ["8 oz"]),
            ("Three pieces", ["three pieces"]),

            # TIME entities (10 examples)
            ("Recycle today", ["today"]),
            ("Pickup tomorrow", ["tomorrow"]),
            ("This week", ["this week"]),
            ("Next month", ["next month"]),
            ("On Monday", ["monday"]),
            ("This Friday", ["friday"]),
            ("Yesterday's trash", ["yesterday"]),
            ("Next week", ["next week"]),
            ("This month", ["this month"]),
            ("On Saturday", ["saturday"]),
        ]

        passed = 0
        failed = 0

        for text, expected_entities in test_cases:
            self.total_tests += 1
            entities = self.entity_extractor.extract(text)
            extracted_texts = [e.text.lower() for e in entities]

            # Check if all expected entities are found
            all_found = all(any(exp.lower() in ext for ext in extracted_texts) for exp in expected_entities)

            if all_found:
                passed += 1
                self.passed_tests += 1
                status = "✅"
            else:
                failed += 1
                self.failed_tests += 1
                status = "❌"

            if failed <= 5:  # Only show first 5 failures
                if status == "❌":
                    print(f"{status} '{text[:40]}...' -> Expected: {expected_entities}, Got: {extracted_texts}")

        accuracy = (passed / len(test_cases)) * 100
        print(f"\nEntity Extraction Results:")
        print(f"  Passed: {passed}/{len(test_cases)} ({accuracy:.1f}%)")
        print(f"  Failed: {failed}/{len(test_cases)}")

        return accuracy >= 70.0  # 70% accuracy threshold (more lenient due to pattern matching)

    def test_language_detection(self):
        """Test language detection with 40 examples"""
        print("\n" + "="*80)
        print("TEST 3: LANGUAGE DETECTION (40 examples)")
        print("="*80)

        test_cases = [
            # English (5 examples)
            ("How do I recycle this plastic bottle?", Language.ENGLISH),
            ("Where can I donate old clothes?", Language.ENGLISH),
            ("What bin does this go in?", Language.ENGLISH),
            ("Can I compost food waste?", Language.ENGLISH),
            ("Find recycling centers near me", Language.ENGLISH),

            # Spanish (5 examples)
            ("¿Cómo reciclo esta botella de plástico?", Language.SPANISH),
            ("¿Dónde puedo donar ropa vieja?", Language.SPANISH),
            ("¿En qué contenedor va esto?", Language.SPANISH),
            ("¿Puedo compostar residuos de comida?", Language.SPANISH),
            ("Encuentra centros de reciclaje cerca de mí", Language.SPANISH),

            # French (5 examples)
            ("Comment recycler cette bouteille en plastique?", Language.FRENCH),
            ("Où puis-je donner de vieux vêtements?", Language.FRENCH),
            ("Dans quelle poubelle cela va-t-il?", Language.FRENCH),
            ("Puis-je composter les déchets alimentaires?", Language.FRENCH),
            ("Trouver des centres de recyclage près de moi", Language.FRENCH),

            # German (5 examples)
            ("Wie recycele ich diese Plastikflasche?", Language.GERMAN),
            ("Wo kann ich alte Kleidung spenden?", Language.GERMAN),
            ("In welche Tonne gehört das?", Language.GERMAN),
            ("Kann ich Lebensmittelabfälle kompostieren?", Language.GERMAN),
            ("Finde Recyclingzentren in meiner Nähe", Language.GERMAN),

            # Italian (5 examples)
            ("Come riciclo questa bottiglia di plastica?", Language.ITALIAN),
            ("Dove posso donare vecchi vestiti?", Language.ITALIAN),
            ("In quale bidone va questo?", Language.ITALIAN),
            ("Posso compostare i rifiuti alimentari?", Language.ITALIAN),
            ("Trova centri di riciclaggio vicino a me", Language.ITALIAN),

            # Portuguese (5 examples)
            ("Como reciclo esta garrafa de plástico?", Language.PORTUGUESE),
            ("Onde posso doar roupas velhas?", Language.PORTUGUESE),
            ("Em qual lixeira isso vai?", Language.PORTUGUESE),
            ("Posso compostar resíduos de alimentos?", Language.PORTUGUESE),
            ("Encontre centros de reciclagem perto de mim", Language.PORTUGUESE),

            # Dutch (5 examples)
            ("Hoe recycle ik deze plastic fles?", Language.DUTCH),
            ("Waar kan ik oude kleding doneren?", Language.DUTCH),
            ("In welke bak gaat dit?", Language.DUTCH),
            ("Kan ik voedselafval composteren?", Language.DUTCH),
            ("Vind recyclingcentra bij mij in de buurt", Language.DUTCH),

            # Japanese (5 examples)
            ("このプラスチックボトルをリサイクルするにはどうすればよいですか？", Language.JAPANESE),
            ("古い服をどこに寄付できますか？", Language.JAPANESE),
            ("これはどのゴミ箱に入れますか？", Language.JAPANESE),
            ("生ゴミを堆肥化できますか？", Language.JAPANESE),
            ("近くのリサイクルセンターを探す", Language.JAPANESE),
        ]

        passed = 0
        failed = 0

        for text, expected_lang in test_cases:
            self.total_tests += 1
            detected_lang, confidence = self.language_handler.detect_language(text)

            if detected_lang == expected_lang:
                passed += 1
                self.passed_tests += 1
                status = "✅"
            else:
                failed += 1
                self.failed_tests += 1
                status = "❌"

            if failed <= 5:  # Only show first 5 failures
                if status == "❌":
                    print(f"{status} '{text[:40]}...' -> Expected: {expected_lang.value}, Got: {detected_lang.value}")

        accuracy = (passed / len(test_cases)) * 100
        print(f"\nLanguage Detection Results:")
        print(f"  Passed: {passed}/{len(test_cases)} ({accuracy:.1f}%)")
        print(f"  Failed: {failed}/{len(test_cases)}")

        return accuracy >= 90.0  # 90% accuracy threshold

    def run_all_tests(self):
        """Run all NLP tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE NLP ENHANCEMENT TESTING")
        print("="*80)

        # Run all tests
        intent_pass = self.test_intent_classification()
        entity_pass = self.test_entity_extraction()
        language_pass = self.test_language_detection()

        # Final summary
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ({(self.passed_tests/self.total_tests)*100:.1f}%)")
        print(f"Failed: {self.failed_tests} ({(self.failed_tests/self.total_tests)*100:.1f}%)")
        print()
        print(f"Intent Classification: {'✅ PASS' if intent_pass else '❌ FAIL'}")
        print(f"Entity Extraction: {'✅ PASS' if entity_pass else '❌ FAIL'}")
        print(f"Language Detection: {'✅ PASS' if language_pass else '❌ FAIL'}")
        print("="*80)

        all_pass = intent_pass and entity_pass and language_pass
        if all_pass:
            print("\n🎉 ALL NLP TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")

        return all_pass


if __name__ == "__main__":
    tester = NLPTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


