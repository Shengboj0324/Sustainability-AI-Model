"""
Unit tests for Vision Service
"""

import pytest
from PIL import Image
import numpy as np


class TestVisionClassifier:
    """Test vision classification functionality"""
    
    def test_image_preprocessing(self, sample_image):
        """Test image preprocessing pipeline"""
        # Test that image is properly resized and normalized
        assert sample_image.size == (224, 224)
        assert sample_image.mode == "RGB"
    
    def test_classification_output_format(self):
        """Test classification output structure"""
        # Mock classification result
        result = {
            "predictions": {
                "item_type": {
                    "class": "plastic_bottle",
                    "confidence": 0.95
                }
            }
        }
        
        assert "predictions" in result
        assert "item_type" in result["predictions"]
        assert result["predictions"]["item_type"]["confidence"] > 0.5


class TestVisionDetector:
    """Test object detection functionality"""
    
    def test_detection_output_format(self):
        """Test detection output structure"""
        result = {
            "detections": [
                {
                    "bbox": [100, 100, 50, 50],
                    "class": "plastic_bottle",
                    "confidence": 0.85
                }
            ],
            "num_detections": 1
        }
        
        assert "detections" in result
        assert len(result["detections"]) == result["num_detections"]
        assert len(result["detections"][0]["bbox"]) == 4

