from .tenons.backbones import *
from .tenons.spatial_temporal_modules import *
from .tenons.segmental_consensuses import *
from .tenons.cls_heads import *
from .tenons.task_heads import *
from .recognizers import *
from .tenons.necks import *
#from .tenons.roi_extractors import *
#from .tenons.anchor_heads import *
from .tenons.shared_heads import *
#from .tenons.bbox_heads import *
from .tenons.discriminators import *
#from .detectors import *
#from .localizers import *


from .registry import (BACKBONES, SPATIAL_TEMPORAL_MODULES, SEGMENTAL_CONSENSUSES, HEADS,
                       RECOGNIZERS, LOCALIZERS, DETECTORS, ARCHITECTURES,
                       NECKS,
                       #ROI_EXTRACTORS,
                       DISCRIMINATORS)
from .builder import (build_backbone, build_spatial_temporal_module, build_segmental_consensus,
                      build_head, build_recognizer, build_detector,
                      build_localizer, build_architecture,
                      build_neck,
                      #build_roi_extractor,
                      build_discriminator)

__all__ = [
    'BACKBONES', 'SPATIAL_TEMPORAL_MODULES', 'SEGMENTAL_CONSENSUSES', 'HEADS',
    'RECOGNIZERS', 'LOCALIZERS', 'DETECTORS', 'ARCHITECTURES',
    'NECKS', #'ROI_EXTRACTORS',
    'build_backbone', 'build_spatial_temporal_module', 'build_segmental_consensus',
    'build_head', 'build_recognizer', 'build_detector',
    'build_localizer', 'build_architecture',
    'build_neck', #'build_roi_extractor'
]
