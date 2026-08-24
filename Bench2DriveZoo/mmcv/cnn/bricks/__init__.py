from mmcv.models.bricks import *  # noqa: F401,F403

from .drop import *  # noqa: F401,F403
from .registry import *  # noqa: F401,F403
from .transformer import *  # noqa: F401,F403

# shim: DepthwiseSeparableConvModule needed by system mmdet
import warnings
from mmcv.cnn import ConvModule

class DepthwiseSeparableConvModule(ConvModule):
    """Depthwise separable convolution shim (mmcv 1.7+ class)."""
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        kwargs.setdefault('groups', in_channels)
        super().__init__(in_channels, out_channels, *args, **kwargs)
