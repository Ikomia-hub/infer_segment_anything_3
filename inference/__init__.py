"""
Inference module for SAM3.
"""
from infer_segment_anything_3.inference.sam3_inference import (
    infer_text_predictor,
    infer_geometric_predictor
)

__all__ = ['infer_text_predictor', 'infer_geometric_predictor']
