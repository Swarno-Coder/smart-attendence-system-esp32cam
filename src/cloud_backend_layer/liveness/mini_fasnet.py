"""
Enhanced Liveness Detection Module
===================================
Multi-factor anti-spoofing using:
1. MiniFASNet deep learning model
2. Print Detection (detect unnatural sharpness, color uniformity)
3. Reflection/Texture Analysis
4. Combined scoring

Key insight: Printed photos often appear SHARPER and MORE UNIFORM than real faces.
Real faces from ESP32-CAM have natural noise and slight blur.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
import logging

from .MiniFASNet import MiniFASNetV2, MiniFASNetV1, MiniFASNetV2SE, MiniFASNetV1SE

logger = logging.getLogger(__name__)


class PrintDetector:
    """
    Detects printed photos based on empirically-derived characteristics:
    
    Real ESP32-CAM faces:
    - laplacian_var: 170-430
    - edge_density: 0.04-0.06  
    - gradient_mean: 24-30
    - hsv_v_var: 278-361
    
    Printed/Screenshot photos:
    - laplacian_var: 450-1000+ (MUCH sharper)
    - edge_density: 0.10-0.17 (2-3x more edges)
    - gradient_mean: 48-67 (2x higher gradients)
    - hsv_v_var: 1500-4200+ (lighting on paper/screen)
    """
    
    def analyze(self, face_image: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze if image appears to be a printed photo.
        
        Returns:
            score: 0.0 (definitely print) to 1.0 (likely real)
        """
        if face_image is None or face_image.size == 0:
            return 0.0, {"error": "Invalid image"}
        
        # Convert and resize to standard size
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(face_image, (128, 128))
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        else:
            gray = face_image
            hsv = None
        gray = cv2.resize(gray, (128, 128))
        
        # 1. Laplacian variance (sharpness)
        # Real: 170-430, Print: 450-1000+
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        if laplacian_var < 150:
            sharpness_score = 0.5  # Too blurry - uncertain
        elif laplacian_var < 450:
            sharpness_score = 1.0  # Good range - likely REAL
        elif laplacian_var < 700:
            sharpness_score = 0.3  # Too sharp - likely PRINT
        else:
            sharpness_score = 0.1  # Very sharp - definitely PRINT
        
        # 2. Edge density
        # Real: 0.04-0.06, Print: 0.10-0.17
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.03:
            edge_score = 0.5  # Very few edges - uncertain
        elif edge_density < 0.08:
            edge_score = 1.0  # Normal - likely REAL
        elif edge_density < 0.12:
            edge_score = 0.4  # High edges - likely PRINT
        else:
            edge_score = 0.1  # Very high edges - definitely PRINT
        
        # 3. Gradient mean
        # Real: 24-30, Print: 48-67
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = gradient.mean()
        
        if gradient_mean < 20:
            gradient_score = 0.5  # Very low - uncertain
        elif gradient_mean < 40:
            gradient_score = 1.0  # Normal - likely REAL
        elif gradient_mean < 55:
            gradient_score = 0.4  # High - likely PRINT
        else:
            gradient_score = 0.1  # Very high - definitely PRINT
        
        # 4. HSV Value variance (lighting uniformity)
        # Real: 278-361, Print: 1500-4200+
        hsv_v_score = 0.5
        if hsv is not None:
            hsv_v_var = hsv[:,:,2].var()
            
            if hsv_v_var < 250:
                hsv_v_score = 0.5  # Very uniform - uncertain
            elif hsv_v_var < 500:
                hsv_v_score = 1.0  # Normal - likely REAL
            elif hsv_v_var < 1200:
                hsv_v_score = 0.6  # Moderate variance
            else:
                hsv_v_score = 0.2  # Very high variance - likely PRINT
        else:
            hsv_v_var = 0
        
        # Weighted combination with emphasis on strongest discriminators
        final_score = (
            0.30 * sharpness_score +
            0.25 * edge_score +
            0.25 * gradient_score +
            0.20 * hsv_v_score
        )
        
        details = {
            "laplacian_var": laplacian_var,
            "sharpness_score": sharpness_score,
            "edge_density": edge_density,
            "edge_score": edge_score,
            "gradient_mean": gradient_mean,
            "gradient_score": gradient_score,
            "hsv_v_var": hsv_v_var if hsv is not None else 0,
            "hsv_v_score": hsv_v_score,
            "final_score": final_score
        }
        
        return final_score, details


class ReflectionAnalyzer:
    """
    Analyzes reflection patterns to detect live faces vs prints.
    Real faces have natural specular highlights; prints have flat lighting.
    """
    
    def analyze(self, face_image: np.ndarray) -> Tuple[float, Dict]:
        """Analyze reflection patterns."""
        if face_image is None or face_image.size == 0:
            return 0.0, {"error": "Invalid image"}
        
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        gray = cv2.resize(gray, (128, 128))
        
        # 1. Specular highlight detection
        # Real faces have subtle specular highlights on nose, forehead, cheeks
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        highlight_ratio = np.sum(bright_mask > 0) / bright_mask.size
        
        # Empirical: Real ESP32-CAM photos have ~0% highlights
        # Printed photos have 4-24% highlights (from screen/paper reflection)
        if highlight_ratio < 0.02:
            highlight_score = 1.0  # Minimal highlights - likely REAL
        elif highlight_ratio < 0.08:
            highlight_score = 0.5  # Some highlights - uncertain
        else:
            highlight_score = 0.2  # Too many highlights - likely PRINT
        
        # 2. Dynamic range
        # Real faces photographed have natural dynamic range
        min_val, max_val = gray.min(), gray.max()
        dynamic_range = max_val - min_val
        
        if dynamic_range < 100:
            range_score = 0.3  # Low contrast - suspicious
        elif dynamic_range < 200:
            range_score = 0.8
        else:
            range_score = 1.0  # Good dynamic range
        
        # 3. Gradient distribution (edge smoothness)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Check for unnatural sharp edges
        sharp_edges = np.sum(gradient > 100) / gradient.size
        if sharp_edges > 0.15:
            gradient_score = 0.4  # Too many sharp edges - paper edge?
        else:
            gradient_score = 0.9
        
        final_score = (
            0.40 * highlight_score +
            0.30 * range_score +
            0.30 * gradient_score
        )
        
        return final_score, {
            "highlight_ratio": highlight_ratio,
            "highlight_score": highlight_score,
            "dynamic_range": dynamic_range,
            "range_score": range_score,
            "sharp_edges": sharp_edges,
            "gradient_score": gradient_score,
            "final_score": final_score
        }


class EnhancedLivenessDetector:
    """
    Multi-factor liveness detection combining:
    1. MiniFASNet deep learning model
    2. Print detection (sharpness, color uniformity)
    3. Reflection analysis
    """
    
    def __init__(self, model_path: str, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        
        # Initialize analyzers
        self.print_detector = PrintDetector()
        self.reflection_analyzer = ReflectionAnalyzer()
        
        # Weights for score fusion
        # Since MiniFASNet isn't reliable alone, weight classical methods more
        self.weights = {
            "minifas": 0.25,
            "print": 0.40,
            "reflection": 0.35
        }
        
        # Threshold for live/fake decision
        # Set higher to ensure printed photos are rejected
        self.threshold = 0.65
        
        # Preprocessing
        self.input_size = (80, 80)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        try:
            self._load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load MiniFASNet: {e}")
    
    def _load_model(self, model_path: str):
        """Load MiniFASNet model."""
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        for create_model in [
            lambda: MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5), num_classes=3),
            lambda: MiniFASNetV1SE(embedding_size=128, conv6_kernel=(5, 5), num_classes=3),
        ]:
            try:
                self.model = create_model()
                self.model.load_state_dict(new_state_dict, strict=True)
                self.model.to(self.device)
                self.model.eval()
                self.model_loaded = True
                logger.info("MiniFASNet loaded")
                return
            except:
                continue
    
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_float = face_rgb.astype(np.float32) / 255.0
        face_normalized = (face_float - self.mean) / self.std
        return torch.from_numpy(face_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def _predict_minifas(self, face_image: np.ndarray) -> float:
        if not self.model_loaded:
            return 0.5
        try:
            input_tensor = self.preprocess(face_image)
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            return probs[0, 1:].sum().item() if probs.shape[1] == 3 else probs[0, 1].item()
        except:
            return 0.5
    
    def predict(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Liveness prediction using MiniFASNetV2 only.
        
        NOTE: Advanced print detection and reflection analysis are commented out
        for now to verify recognition works. Uncomment later for full anti-spoofing.
        
        Returns:
            (is_live, confidence)
        """
        if face_image is None or face_image.size == 0:
            return False, 0.0
        
        # Get MiniFASNet deep learning score
        minifas_score = self._predict_minifas(face_image)
        
        # ========== ADVANCED LIVENESS CHECKS (COMMENTED FOR DEBUGGING) ==========
        # TODO: Uncomment these for full anti-spoofing after recognition is verified
        #
        # print_score, _ = self.print_detector.analyze(face_image)
        # reflection_score, _ = self.reflection_analyzer.analyze(face_image)
        # 
        # # Weighted fusion
        # final_score = (
        #     self.weights["minifas"] * minifas_score +
        #     self.weights["print"] * print_score +
        #     self.weights["reflection"] * reflection_score
        # )
        # 
        # # VETO LOGIC: If print detector confidently identifies a print, reject
        # # Print score < 0.6 indicates strong print characteristics
        # if print_score < 0.6:
        #     is_live = False
        #     confidence = 0.5 + (0.6 - print_score)  # Higher confidence for lower print scores
        #     confidence = min(1.0, confidence)
        # else:
        #     is_live = final_score > self.threshold
        #     confidence = 0.5 + abs(final_score - self.threshold)
        #     confidence = min(1.0, confidence)
        # 
        # logger.debug(f"Liveness: minifas={minifas_score:.2f}, print={print_score:.2f}, "
        #             f"reflection={reflection_score:.2f}, final={final_score:.2f}")
        # =========================================================================
        
        # Simple MiniFASNet-only liveness check
        # Score > 0.5 means the model thinks it's a real face
        is_live = minifas_score > 0.5
        confidence = minifas_score
        
        logger.debug(f"Liveness (MiniFASNet only): score={minifas_score:.2f}")
        
        return is_live, confidence


# Backward compatible alias
class LivenessPredictor(EnhancedLivenessDetector):
    pass
