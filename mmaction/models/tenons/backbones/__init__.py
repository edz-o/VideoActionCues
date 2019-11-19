from .bninception import BNInception
from .resnet import ResNet

from .inception_v1_i3d import InceptionV1_I3D
from .resnet_i3d import ResNet_I3D
#from .resnet_s3d import ResNet_S3D
from .inception_i3d_yi import I3D

__all__ = [
    'BNInception',
    'ResNet',
    'InceptionV1_I3D',
    'ResNet_I3D',
    #'ResNet_S3D',
    'I3D',
]
