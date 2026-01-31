"""
Liveness Detection Module
=========================
Provides face anti-spoofing using MiniFASNetV2.
"""

from .mini_fasnet import MiniFASNetV2, MiniFASNetV2SE, LivenessPredictor

__all__ = ['MiniFASNetV2', 'MiniFASNetV2SE', 'LivenessPredictor']
