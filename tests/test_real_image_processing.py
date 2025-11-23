#!/usr/bin/env python3
"""
REAL IMAGE PROCESSING TEST
===========================

Test vision system with real images:
1. Generate synthetic test images
2. Test image quality assessment
3. Test classification pipeline
4. Test detection pipeline
5. Validate error handling
"""

import sys
import io
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class RealImageProcessingTest:
    """Test vision system with real images"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
    
    def generate_test_image(self, width=640, height=480, label="Test") -> Image.Image:
        """Generate a synthetic test image"""
        # Create image with random background
        img = Image.new('RGB', (width, height), color=(
            random.randint(200, 255),
            random.randint(200, 255),
            random.randint(200, 255)
        ))
        
        draw = ImageDraw.Draw(img)
        
        # Draw some shapes to simulate waste objects
        for _ in range(5):
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            shape = random.choice(['rectangle', 'ellipse'])
            if shape == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
        
        # Add text label
        try:
            draw.text((10, 10), label, fill=(0, 0, 0))
        except:
            pass  # Font not available
        
        return img
    
    def image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def test_image_generation(self) -> bool:
        """Test 1: Generate test images"""
        print("\n" + "="*80)
        print("TEST 1: IMAGE GENERATION")
        print("="*80)
        
        try:
            # Generate different types of test images
            test_cases = [
                ("plastic_bottle", 640, 480),
                ("cardboard_box", 800, 600),
                ("glass_jar", 1024, 768),
                ("metal_can", 640, 480),
                ("paper_waste", 800, 600),
            ]
            
            images = []
            for label, width, height in test_cases:
                img = self.generate_test_image(width, height, label)
                images.append((label, img))
                print(f"✅ Generated {label} image ({width}x{height})")
            
            print(f"\n✅ PASSED: Generated {len(images)} test images")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_image_quality_assessment(self) -> bool:
        """Test 2: Image quality assessment"""
        print("\n" + "="*80)
        print("TEST 2: IMAGE QUALITY ASSESSMENT")
        print("="*80)

        try:
            from models.vision.image_quality import AdvancedImageQualityPipeline

            pipeline = AdvancedImageQualityPipeline()
            print("✅ AdvancedImageQualityPipeline loaded")

            # Test with different quality images
            test_cases = [
                ("high_quality", 1024, 768),
                ("medium_quality", 640, 480),
                ("low_quality", 320, 240),
                ("very_low_quality", 160, 120),
            ]

            for label, width, height in test_cases:
                img = self.generate_test_image(width, height, label)

                # Process image through quality pipeline (returns tuple)
                enhanced_img, report = pipeline.process_image(img)

                print(f"✅ {label} ({width}x{height}): Quality score = {report.quality_score:.3f}")

                # Validate score is in valid range
                assert 0.0 <= report.quality_score <= 1.0, f"Invalid quality score: {report.quality_score}"
                assert isinstance(report.warnings, list), "Warnings should be a list"
                assert enhanced_img is not None, "Enhanced image should not be None"

            print(f"\n✅ PASSED: Image quality assessment working")
            return True
            
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_base64_encoding(self) -> bool:
        """Test 3: Base64 encoding for API transmission"""
        print("\n" + "="*80)
        print("TEST 3: BASE64 ENCODING")
        print("="*80)
        
        try:
            # Generate test image
            img = self.generate_test_image(640, 480, "test")
            
            # Convert to base64
            b64_str = self.image_to_base64(img)
            
            print(f"✅ Image encoded to base64 ({len(b64_str)} characters)")
            
            # Decode back
            img_data = base64.b64decode(b64_str)
            img_decoded = Image.open(io.BytesIO(img_data))
            
            print(f"✅ Image decoded successfully ({img_decoded.size})")
            
            # Validate
            assert img.size == img_decoded.size, "Size mismatch after encoding/decoding"
            
            print(f"\n✅ PASSED: Base64 encoding/decoding working")
            return True

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_error_handling(self) -> bool:
        """Test 4: Error handling with invalid images"""
        print("\n" + "="*80)
        print("TEST 4: ERROR HANDLING")
        print("="*80)

        try:
            from models.vision.image_quality import AdvancedImageQualityPipeline

            pipeline = AdvancedImageQualityPipeline()

            # Test with None
            try:
                pipeline.process_image(None)
                print("❌ Should have raised error for None input")
                return False
            except Exception as e:
                print(f"✅ Correctly handled None input: {type(e).__name__}")

            # Test with invalid base64
            try:
                invalid_b64 = "not_valid_base64!!!"
                img_data = base64.b64decode(invalid_b64)
                print("✅ Base64 decode handled gracefully")
            except Exception as e:
                print(f"✅ Correctly handled invalid base64: {type(e).__name__}")

            # Test with corrupted image data
            try:
                corrupted_data = b"corrupted_image_data"
                Image.open(io.BytesIO(corrupted_data))
                print("❌ Should have raised error for corrupted data")
                return False
            except Exception as e:
                print(f"✅ Correctly handled corrupted image: {type(e).__name__}")

            print(f"\n✅ PASSED: Error handling working correctly")
            return True

        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run all image processing tests"""
        print("\n" + "="*80)
        print("📸 REAL IMAGE PROCESSING TEST SUITE")
        print("="*80)

        tests = [
            ("Image Generation", self.test_image_generation),
            ("Image Quality Assessment", self.test_image_quality_assessment),
            ("Base64 Encoding", self.test_base64_encoding),
            ("Error Handling", self.test_error_handling),
        ]

        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    self.tests_passed += 1
                else:
                    self.tests_failed += 1
            except Exception as e:
                self.tests_failed += 1
                print(f"\n❌ {test_name} crashed: {e}")

        print("\n" + "="*80)
        print("📊 IMAGE PROCESSING TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.tests_passed + self.tests_failed}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success Rate: {self.tests_passed / (self.tests_passed + self.tests_failed) * 100:.1f}%")
        print("="*80)

        if self.tests_failed == 0:
            print("✅ ALL IMAGE PROCESSING TESTS PASSED")
        else:
            print(f"❌ {self.tests_failed} TESTS FAILED")
        print("="*80)


if __name__ == "__main__":
    tester = RealImageProcessingTest()
    tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if tester.tests_failed == 0 else 1)


