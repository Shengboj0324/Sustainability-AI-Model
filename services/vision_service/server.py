"""
Vision Service - Waste recognition and classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import httpx
import logging
import yaml
import timm
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ReleAF AI Vision Service",
    description="Waste recognition and classification service",
    version="0.1.0"
)


class ClassifyRequest(BaseModel):
    """Classification request"""
    image: Optional[str] = None
    image_url: Optional[str] = None
    return_probabilities: bool = True
    top_k: int = 3


class DetectRequest(BaseModel):
    """Detection request"""
    image: Optional[str] = None
    image_url: Optional[str] = None
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100


class VisionService:
    """Vision inference service"""
    
    def __init__(self, config_path: str = "configs/vision_cls.yaml"):
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self.detector = None
        self.transform = None
        self.item_classes = []
        self.material_classes = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config not found: {config_path}")
            return {}
    
    def load_models(self):
        """Load classifier and detector models"""
        logger.info("Loading vision models...")
        
        # Load classifier
        self._load_classifier()
        
        # Load detector
        self._load_detector()
        
        # Setup transforms
        self._setup_transforms()
        
        logger.info("Vision models loaded successfully")
    
    def _load_classifier(self):
        """Load classification model"""
        try:
            model_name = self.config["model"]["backbone"]
            num_classes_item = self.config["model"]["num_classes_item"]
            
            # Load pretrained model
            self.classifier = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes_item
            )
            
            # Load fine-tuned weights if available
            checkpoint_path = "models/vision/classifier/best_model.pth"
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.classifier.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded classifier checkpoint from {checkpoint_path}")
            except FileNotFoundError:
                logger.warning("No classifier checkpoint found, using pretrained weights")
            
            self.classifier.to(self.device)
            self.classifier.eval()
            
            # Load class names
            self.item_classes = self.config["model"]["item_classes"]
            self.material_classes = self.config["model"]["material_classes"]
            
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise
    
    def _load_detector(self):
        """Load detection model"""
        try:
            # Load YOLO model
            model_path = "models/vision/detector/best.pt"
            try:
                self.detector = YOLO(model_path)
                logger.info(f"Loaded detector from {model_path}")
            except FileNotFoundError:
                # Use pretrained YOLO as fallback
                self.detector = YOLO("yolov8m.pt")
                logger.warning("No detector checkpoint found, using pretrained YOLOv8")
            
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            raise
    
    def _setup_transforms(self):
        """Setup image transforms"""
        input_size = self.config["data"]["input_size"]
        mean = self.config["data"]["mean"]
        std = self.config["data"]["std"]
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    async def load_image(self, image_b64: Optional[str], image_url: Optional[str]) -> Image.Image:
        """Load image from base64 or URL"""
        if image_b64:
            # Decode base64
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif image_url:
            # Download from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            raise ValueError("Either image or image_url must be provided")
        
        return image
    
    @torch.inference_mode()
    def classify(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        """Classify image"""
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.item_classes)))
        
        predictions = {
            "item_type": {
                "class": self.item_classes[top_indices[0][0].item()],
                "confidence": top_probs[0][0].item(),
                "top_k": [
                    {
                        "class": self.item_classes[idx.item()],
                        "confidence": prob.item()
                    }
                    for prob, idx in zip(top_probs[0], top_indices[0])
                ]
            }
        }
        
        return predictions
    
    def detect(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        # Run detection
        results = self.detector(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "bbox": box.xywh[0].tolist(),
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                }
                detections.append(detection)
        
        return detections


# Initialize service
vision_service = VisionService()


@app.on_event("startup")
async def startup():
    """Load models on startup"""
    vision_service.load_models()


@app.post("/classify")
async def classify(request: ClassifyRequest):
    """Classify image"""
    try:
        # Load image
        image = await vision_service.load_image(request.image, request.image_url)
        
        # Classify
        predictions = vision_service.classify(image, request.top_k)
        
        return {
            "predictions": predictions,
            "processing_time_ms": 0  # TODO: track actual time
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect")
async def detect(request: DetectRequest):
    """Detect objects"""
    try:
        # Load image
        image = await vision_service.load_image(request.image, request.image_url)
        
        # Detect
        detections = vision_service.detect(
            image,
            request.conf_threshold,
            request.iou_threshold
        )
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "processing_time_ms": 0
        }
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_and_classify")
async def detect_and_classify(request: DetectRequest):
    """Detect and classify objects"""
    # First detect
    detect_result = await detect(request)
    
    # Then classify each detection (simplified - would crop and classify each box)
    return detect_result


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "vision",
        "models_loaded": vision_service.classifier is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)

