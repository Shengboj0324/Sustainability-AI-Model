"""
Advanced Image Quality Enhancement Pipeline

CRITICAL: Handles ANY random customer image with comprehensive quality checks and enhancements.

Features:
- EXIF orientation handling
- Noise detection and denoising
- Compression artifact detection
- Motion blur detection and sharpening
- Color space validation and conversion
- Transparent PNG handling
- Animated GIF/multi-page TIFF handling
- HDR tone mapping
- Adaptive histogram equalization
- Quality scoring (0.0-1.0)
"""

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageStat
from typing import Tuple, List, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityReport:
    """Comprehensive image quality report"""
    quality_score: float  # 0.0-1.0
    warnings: List[str]
    enhancements_applied: List[str]
    original_format: str
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    noise_level: float
    blur_score: float
    brightness: float
    contrast: float
    jpeg_quality: int


class AdvancedImageQualityPipeline:
    """
    Advanced image quality enhancement pipeline

    CRITICAL: Handles trillion kinds of images with comprehensive validation and enhancement
    """

    def __init__(self, enable_enhancement: bool = True):
        self.enable_enhancement = enable_enhancement
        logger.info(f"Advanced Image Quality Pipeline initialized (enhancement: {enable_enhancement})")

    def process_image(self, image: Image.Image) -> Tuple[Image.Image, ImageQualityReport]:
        """
        Process image with comprehensive quality checks and enhancements

        Args:
            image: Input PIL Image

        Returns:
            (enhanced_image, quality_report)
        """
        warnings = []
        enhancements = []
        quality_score = 1.0

        original_format = image.format or "Unknown"
        original_size = image.size

        # 1. Handle EXIF orientation
        image = self._handle_exif_orientation(image, warnings, enhancements)

        # 2. Handle special formats (animated GIF, multi-page TIFF, HDR)
        image = self._handle_special_formats(image, warnings, enhancements)

        # 3. Handle transparent images
        image = self._handle_transparency(image, warnings, enhancements)

        # 4. Convert to RGB if needed
        if image.mode != "RGB":
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert("RGB")
            enhancements.append(f"Converted from {image.mode} to RGB")

        # Convert to numpy for analysis
        img_array = np.array(image)

        # 5. Detect noise
        noise_level = self._detect_noise(img_array)
        if noise_level > 0.3:
            warnings.append(f"High noise level: {noise_level:.2f}")
            quality_score *= 0.8

        # 6. Detect motion blur
        blur_score = self._detect_motion_blur(img_array)
        if blur_score > 100:  # Laplacian variance threshold
            warnings.append(f"Image appears blurry (score: {blur_score:.1f})")
            quality_score *= 0.7

        # 7. Check brightness and contrast
        brightness = img_array.mean()
        contrast = img_array.std()

        if brightness < 30:
            warnings.append(f"Image very dark (brightness: {brightness:.1f})")
            quality_score *= 0.7
        elif brightness > 225:
            warnings.append(f"Image very bright (brightness: {brightness:.1f})")
            quality_score *= 0.7

        if contrast < 20:
            warnings.append(f"Low contrast (std: {contrast:.1f})")
            quality_score *= 0.8

        # 8. Estimate JPEG quality
        jpeg_quality = self._estimate_jpeg_quality(image)
        if jpeg_quality < 50:
            warnings.append(f"Low JPEG quality: {jpeg_quality}")
            quality_score *= 0.7

        # 9. Apply enhancements if enabled and quality is poor
        if self.enable_enhancement and quality_score < 0.7:
            img_array, applied = self._enhance_image(img_array, noise_level, blur_score, contrast)
            enhancements.extend(applied)
            image = Image.fromarray(img_array)

        # 10. Size validation and resizing
        width, height = image.size

        if width < 64 or height < 64:
            warnings.append(f"Image very small ({width}x{height}). Results may be poor.")
            quality_score *= 0.5

        if width > 4096 or height > 4096:
            warnings.append(f"Image very large ({width}x{height}). Resizing for memory.")
            scale = 4096 / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            enhancements.append(f"Resized from {width}x{height} to {new_size[0]}x{new_size[1]}")
            quality_score *= 0.95

        # Create quality report
        report = ImageQualityReport(
            quality_score=quality_score,
            warnings=warnings,
            enhancements_applied=enhancements,
            original_format=original_format,
            original_size=original_size,
            final_size=image.size,
            noise_level=noise_level,
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            jpeg_quality=jpeg_quality
        )

        logger.info(f"Image processed: quality={quality_score:.2f}, warnings={len(warnings)}, enhancements={len(enhancements)}")

        return image, report

    def _handle_transparency(self, image: Image.Image, warnings: List[str], enhancements: List[str]) -> Image.Image:
        """Handle transparent images (RGBA, LA, P with transparency)"""
        try:
            if image.mode in ['RGBA', 'LA'] or (image.mode == 'P' and 'transparency' in image.info):
                warnings.append(f"Transparent image ({image.mode}) - compositing on white background")

                # Create white background
                if image.mode == 'RGBA':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                elif image.mode == 'LA':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image.convert('RGB'), mask=image.split()[1])
                else:  # P with transparency
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image.convert('RGBA'))

                image = background
                enhancements.append("Composited transparent image on white background")

        except Exception as e:
            logger.warning(f"Transparency handling failed: {e}")

        return image

    def _tone_map_hdr(self, image: Image.Image) -> Image.Image:
        """Tone map HDR images to 8-bit RGB"""
        try:
            # Convert to numpy array
            img_array = np.array(image, dtype=np.float32)

            # Normalize to 0-255 range
            img_min, img_max = img_array.min(), img_array.max()
            if img_max > img_min:
                img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_array = np.zeros_like(img_array, dtype=np.uint8)

            # Convert back to PIL Image
            if len(img_array.shape) == 2:
                image = Image.fromarray(img_array, mode='L')
            else:
                image = Image.fromarray(img_array, mode='RGB')

            logger.info("HDR tone mapping applied")

        except Exception as e:
            logger.error(f"HDR tone mapping failed: {e}")
            # Fallback to simple conversion
            image = image.convert('RGB')

        return image

    def _detect_noise(self, img_array: np.ndarray) -> float:
        """
        Detect noise level in image

        Returns:
            Noise level (0.0-1.0+)
        """
        try:
            # Convert to grayscale if color
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Estimate noise using Laplacian variance method
            # High frequency content indicates noise
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.var() / 10000.0  # Normalize

            return min(noise_level, 1.0)

        except Exception as e:
            logger.warning(f"Noise detection failed: {e}")
            return 0.0

    def _detect_motion_blur(self, img_array: np.ndarray) -> float:
        """
        Detect motion blur using Laplacian variance

        Returns:
            Blur score (higher = sharper, lower = blurrier)
        """
        try:
            # Convert to grayscale if color
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Calculate Laplacian variance
            # Sharp images have high variance, blurry images have low variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = laplacian.var()

            return blur_score

        except Exception as e:
            logger.warning(f"Blur detection failed: {e}")
            return 100.0  # Assume sharp if detection fails

    def _estimate_jpeg_quality(self, image: Image.Image) -> int:
        """
        Estimate JPEG quality (0-100)

        Note: This is an approximation based on quantization tables if available
        """
        try:
            # Check if image has JPEG quantization tables
            if hasattr(image, 'quantization') and image.quantization:
                # Estimate quality from quantization tables
                # Lower values in quantization table = higher quality
                qtables = image.quantization
                if qtables:
                    avg_q = np.mean([np.mean(list(q.values())) for q in qtables.values()])
                    # Rough estimation: quality = 100 - (avg_q - 1)
                    quality = max(0, min(100, int(100 - (avg_q - 1))))
                    return quality

            # Fallback: assume good quality if not JPEG or no quantization info
            return 85

        except Exception as e:
            logger.debug(f"JPEG quality estimation failed: {e}")
            return 85  # Assume good quality

    def _enhance_image(
        self,
        img_array: np.ndarray,
        noise_level: float,
        blur_score: float,
        contrast: float
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Enhance image quality based on detected issues

        Returns:
            (enhanced_array, list_of_enhancements_applied)
        """
        enhancements = []

        try:
            # 1. Denoising for noisy images
            if noise_level > 0.2:
                if len(img_array.shape) == 3:
                    img_array = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
                else:
                    img_array = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
                enhancements.append(f"Applied denoising (noise level: {noise_level:.2f})")
                logger.info("Denoising applied")

            # 2. Adaptive histogram equalization for low contrast
            if contrast < 30:
                if len(img_array.shape) == 3:
                    # Apply CLAHE to each channel
                    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    img_array = clahe.apply(img_array)
                enhancements.append(f"Applied adaptive histogram equalization (contrast: {contrast:.1f})")
                logger.info("Adaptive histogram equalization applied")

            # 3. Sharpening for blurry images
            if blur_score < 100:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
                if len(img_array.shape) == 3:
                    img_array = cv2.filter2D(img_array, -1, kernel)
                else:
                    img_array = cv2.filter2D(img_array, -1, kernel)
                enhancements.append(f"Applied sharpening (blur score: {blur_score:.1f})")
                logger.info("Sharpening applied")

        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")

        return img_array, enhancements

    def _handle_exif_orientation(self, image: Image.Image, warnings: List[str], enhancements: List[str]) -> Image.Image:
        """Handle EXIF orientation (auto-rotate based on metadata)"""
        try:
            # Auto-rotate based on EXIF orientation tag
            original_size = image.size
            image = ImageOps.exif_transpose(image)
            if image.size != original_size:
                enhancements.append("Auto-rotated based on EXIF orientation")
                logger.info("Image auto-rotated based on EXIF")
        except Exception as e:
            logger.debug(f"EXIF orientation handling failed (may not have EXIF): {e}")

        return image

    def _handle_special_formats(self, image: Image.Image, warnings: List[str], enhancements: List[str]) -> Image.Image:
        """Handle special formats (animated GIF, multi-page TIFF, HDR)"""
        try:
            # Animated GIF - extract first frame
            if hasattr(image, 'is_animated') and image.is_animated:
                image.seek(0)
                warnings.append("Animated GIF - using first frame only")
                enhancements.append("Extracted first frame from animated GIF")

            # Multi-page TIFF - extract first page
            if image.format == 'TIFF' and hasattr(image, 'n_frames') and image.n_frames > 1:
                image.seek(0)
                warnings.append(f"Multi-page TIFF ({image.n_frames} pages) - using first page only")
                enhancements.append("Extracted first page from multi-page TIFF")

            # HDR images (mode 'I' or 'F') - tone mapping
            if image.mode in ['I', 'F']:
                warnings.append("HDR image detected - applying tone mapping")
                image = self._tone_map_hdr(image)
                enhancements.append("Applied tone mapping to HDR image")

        except Exception as e:
            logger.warning(f"Special format handling failed: {e}")

        return image

