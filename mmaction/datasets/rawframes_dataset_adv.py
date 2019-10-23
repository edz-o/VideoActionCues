import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (GroupImageTransform)
from .utils import to_tensor
from .rawframes_dataset import RawFramesDataset
from .rawframes_dataset_mt import MTRawFramesDataset


class RawFramesDatasetAdv(Dataset):
    def __init__(self,
                 ann_file0,
                 ann_file1,
                 img_prefix0,
                 img_prefix1,
                 img_norm_cfg,
                 num_segments=3,
                 new_length=1,
                 new_step=1,
                 random_shift=True,
                 temporal_jitter=False,
                 modality='RGB',
                 modality2='RGB',
                 image_tmpl0='img_{}.jpg',
                 image_tmpl1='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 div_255=False,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0.5,
                 resize_keep_ratio=True,
                 resize_ratio=[1, 0.875, 0.75, 0.66],
                 test_mode=False,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scales=None,
                 max_distort=1,
                 input_format='NCHW'):

        self.modality = modality
        self.src_data = MTRawFramesDataset(ann_file0, img_prefix0, img_norm_cfg, num_segments,
                            new_length, new_step, random_shift, temporal_jitter,
                            modality, image_tmpl0, img_scale, img_scale_file, input_size,
                            div_255, size_divisor, proposal_file, num_max_proposals, flip_ratio,
                            resize_keep_ratio, resize_ratio, test_mode, oversample, random_crop,
                            more_fix_crop, multiscale_crop, scales, max_distort, input_format)

        if isinstance(modality, (tuple, list)):
            modality = modality[0]
        self.tgt_data = RawFramesDataset(ann_file1, img_prefix1, img_norm_cfg, num_segments,
                            new_length, new_step, random_shift, temporal_jitter,
                            modality2, image_tmpl1, img_scale, img_scale_file, input_size,
                            div_255, size_divisor, proposal_file, num_max_proposals, flip_ratio,
                            resize_keep_ratio, resize_ratio, test_mode, oversample, random_crop,
                            more_fix_crop, multiscale_crop, scales, max_distort, input_format)

        self.test_mode = test_mode

        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            # img_info = self.img_infos[i]
            # if img_info['width'] / img_info['height'] > 1:
            self.flag[i] = 1

    def __getitem__(self, idx):
        ns = len(self.src_data)
        nt = len(self.tgt_data)
        data_src = self.src_data[idx % ns]
        data_tgt = self.tgt_data[idx % nt]
        if 'Seg' in self.modality:
            data = dict(
                    num_modalities=data_src['num_modalities'],
                    img_group_0=data_src['img_group_0'], img_group_1=data_tgt['img_group_0'],
                    img_group_seg=data_src['img_group_1'],
                    gt_label0=data_src['gt_label'], gt_label1=data_tgt['gt_label'],
                    img_meta=data_src['img_meta'])
        else:
            data = dict(
                    num_modalities=data_src['num_modalities'],
                    img_group_0=data_src['img_group_0'], img_group_1=data_tgt['img_group_0'],
                    gt_label0=data_src['gt_label'], gt_label1=data_tgt['gt_label'],
                    img_meta=data_src['img_meta'])

        return data

    def __len__(self):
        return max(len(self.src_data), len(self.tgt_data))


