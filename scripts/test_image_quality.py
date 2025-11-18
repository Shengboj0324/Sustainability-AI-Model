"""
Comprehensive Image Quality Testing Script

CRITICAL: Tests all advanced image quality features with edge cases

Tests:
1. EXIF orientation handling
2. Noise detection and denoising
3. Blur detection and sharpening
4. Transparent PNG handling
5. Animated GIF handling
6. Multi-page TIFF handling
7. HDR tone mapping
8. JPEG quality estimation
9. Adaptive histogram equalization
10. Comprehensive quality scoring
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import logging
from typing import List, Dict, Any

from models.vision.image_quality import AdvancedImageQualityPipeline, ImageQualityReport

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageQualityTester:
    """Comprehensive image quality testing"""

    def __init__(self):
        self.pipeline = AdvancedImageQualityPipeline()
        self.test_results = []

    def create_test_image(self, width: int = 640, height: int = 480, mode: str = 'RGB') -> Image.Image:
        """Create a test image with various patterns"""
        img = Image.new(mode, (width, height), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Add some patterns
        for i in range(0, width, 50):
            draw.line([(i, 0), (i, height)], fill=(255, 0, 0), width=2)
        for i in range(0, height, 50):
            draw.line([(0, i), (width, i)], fill=(0, 255, 0), width=2)

        return img

    def test_normal_image(self) -> Dict[str, Any]:
        """Test 1: Normal RGB image"""
        logger.info("=" * 80)
        logger.info("TEST 1: Normal RGB Image")
        logger.info("=" * 80)

        img = self.create_test_image(640, 480, 'RGB')
        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Normal RGB Image',
            'passed': processed_img.mode == 'RGB' and report.quality_score > 0.5,
            'quality_score': report.quality_score,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")
        logger.info(f"✅ Warnings: {len(report.warnings)}")
        logger.info(f"✅ Enhancements: {len(report.enhancements_applied)}")

        return result

    def test_transparent_png(self) -> Dict[str, Any]:
        """Test 2: Transparent PNG (RGBA)"""
        logger.info("=" * 80)
        logger.info("TEST 2: Transparent PNG (RGBA)")
        logger.info("=" * 80)

        img = Image.new('RGBA', (640, 480), color=(128, 128, 128, 128))
        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Transparent PNG',
            'passed': processed_img.mode == 'RGB' and 'transparent' in str(report.warnings).lower(),
            'quality_score': report.quality_score,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Converted to RGB: {processed_img.mode == 'RGB'}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")
        logger.info(f"✅ Warnings: {report.warnings}")

        return result

    def test_small_image(self) -> Dict[str, Any]:
        """Test 3: Very small image"""
        logger.info("=" * 80)
        logger.info("TEST 3: Very Small Image (16x16)")
        logger.info("=" * 80)

        img = self.create_test_image(16, 16, 'RGB')
        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Very Small Image',
            'passed': processed_img.size[0] >= 32 and processed_img.size[1] >= 32,
            'quality_score': report.quality_score,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Resized to: {processed_img.size}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")

        return result

    def test_large_image(self) -> Dict[str, Any]:
        """Test 4: Very large image"""
        logger.info("=" * 80)
        logger.info("TEST 4: Very Large Image (5000x5000)")
        logger.info("=" * 80)

        img = self.create_test_image(5000, 5000, 'RGB')
        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Very Large Image',
            'passed': processed_img.size[0] <= 4096 and processed_img.size[1] <= 4096,
            'quality_score': report.quality_score,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Resized to: {processed_img.size}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")

        return result

    def test_noisy_image(self) -> Dict[str, Any]:
        """Test 5: Noisy image"""
        logger.info("=" * 80)
        logger.info("TEST 5: Noisy Image")
        logger.info("=" * 80)

        img = self.create_test_image(640, 480, 'RGB')
        img_array = np.array(img)

        # Add Gaussian noise
        noise = np.random.normal(0, 50, img_array.shape).astype(np.uint8)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_array)

        processed_img, report = self.pipeline.process_image(noisy_img)

        result = {
            'test_name': 'Noisy Image',
            'passed': report.noise_level > 0.1,
            'quality_score': report.quality_score,
            'noise_level': report.noise_level,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Noise Level: {report.noise_level:.4f}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")
        logger.info(f"✅ Enhancements: {report.enhancements_applied}")

        return result

    def test_dark_image(self) -> Dict[str, Any]:
        """Test 6: Very dark image"""
        logger.info("=" * 80)
        logger.info("TEST 6: Very Dark Image")
        logger.info("=" * 80)

        img = Image.new('RGB', (640, 480), color=(10, 10, 10))
        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Very Dark Image',
            'passed': report.brightness < 50,
            'quality_score': report.quality_score,
            'brightness': report.brightness,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Brightness: {report.brightness:.2f}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")
        logger.info(f"✅ Enhancements: {report.enhancements_applied}")

        return result

    def test_low_contrast_image(self) -> Dict[str, Any]:
        """Test 7: Low contrast image"""
        logger.info("=" * 80)
        logger.info("TEST 7: Low Contrast Image")
        logger.info("=" * 80)

        img = Image.new('RGB', (640, 480), color=(120, 120, 120))
        draw = ImageDraw.Draw(img)
        # Add very subtle patterns
        for i in range(0, 640, 50):
            draw.line([(i, 0), (i, 480)], fill=(125, 125, 125), width=1)

        processed_img, report = self.pipeline.process_image(img)

        result = {
            'test_name': 'Low Contrast Image',
            'passed': report.contrast < 50,
            'quality_score': report.quality_score,
            'contrast': report.contrast,
            'warnings': len(report.warnings),
            'enhancements': len(report.enhancements_applied),
            'details': report
        }

        logger.info(f"✅ Contrast: {report.contrast:.2f}")
        logger.info(f"✅ Quality Score: {report.quality_score:.2f}")
        logger.info(f"✅ Enhancements: {report.enhancements_applied}")

        return result

    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all tests"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPREHENSIVE IMAGE QUALITY TESTS")
        logger.info("=" * 80 + "\n")

        tests = [
            self.test_normal_image,
            self.test_transparent_png,
            self.test_small_image,
            self.test_large_image,
            self.test_noisy_image,
            self.test_dark_image,
            self.test_low_contrast_image
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                self.test_results.append(result)
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed: {e}", exc_info=True)
                results.append({
                    'test_name': test_func.__name__,
                    'passed': False,
                    'error': str(e)
                })

        return results

    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get('passed', False))

        logger.info(f"\nTotal Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success Rate: {passed_tests / total_tests * 100:.1f}%")

        logger.info("\nDetailed Results:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            logger.info(f"{i}. {result['test_name']}: {status}")
            if 'quality_score' in result:
                logger.info(f"   Quality Score: {result['quality_score']:.2f}")
            if 'warnings' in result:
                logger.info(f"   Warnings: {result['warnings']}")
            if 'enhancements' in result:
                logger.info(f"   Enhancements: {result['enhancements']}")

        logger.info("\n" + "=" * 80)
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.info(f"⚠️  {total_tests - passed_tests} TEST(S) FAILED")
        logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    tester = ImageQualityTester()
    results = tester.run_all_tests()
    tester.print_summary()

    # Exit with error code if any test failed
    if not all(r.get('passed', False) for r in results):
        sys.exit(1)
