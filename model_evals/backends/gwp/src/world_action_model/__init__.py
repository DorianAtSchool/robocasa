"""GigaWorld-Policy world-action model package (inference-only distribution).

Training utilities (trainers, datasets, transforms, pipeline) have been stripped
out. Only the model definition required by ``model_evals/backends/gwp/inference_server.py``
is shipped here.
"""

from .models import CasualWorldActionTransformer

__all__ = ["CasualWorldActionTransformer"]
