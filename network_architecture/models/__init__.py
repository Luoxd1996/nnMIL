"""
nnMIL Models Package
This package contains all MIL model implementations.
"""

# Import all models for convenience
# Note: Only import what exists and is commonly used
from .simple_mil import SimpleMIL
try:
    from .ab_mil import AB_MIL
except ImportError:
    pass

try:
    from .trans_mil import TRANS_MIL
except ImportError:
    pass

try:
    from .wikg_mil import WIKG_MIL
except ImportError:
    pass

try:
    from .ilra_mil import ILRA_MIL
except ImportError:
    pass

try:
    from .ds_mil import DS_MIL
except ImportError:
    pass

try:
    from .dtfd_mil import Attention_with_Classifier
except ImportError:
    pass

try:
    from .vision_transformer import VisionTransformer
except ImportError:
    pass

try:
    from .rrt import RRT
except ImportError:
    pass

__all__ = ['SimpleMIL']
