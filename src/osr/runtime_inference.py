"""Open-Set Runtime Edge Inference.

Corresponds to the runtime inference step:
    Evaluates traffic locally at each edge node in wire-speed time:
    1. Extracts latent mean representation z = mu(x) from frozen encoder.
    2. Evaluates multi-centroid Mahalanobis distances to known classes.
    3. Builds the conformal prediction set C(z).
    4. If C(z) is empty, flags sample as unknown (zero-day intrusion).
    5. If C(z) is non-empty, assigns to candidate class minimizing the normalized distance.
"""

from __future__ import annotations

from src.osr.block_mahalanobis import (
    WireSpeedEdgeDetector,
    MahalanobisInferenceResult,
)

__all__ = [
    "WireSpeedEdgeDetector",
    "MahalanobisInferenceResult",
]
