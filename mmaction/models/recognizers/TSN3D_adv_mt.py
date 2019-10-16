from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from .TSN3D_bb_mt import TSN3D_bb_mt

import torch
import pdb

@RECOGNIZERS.register_module
class TSN3D_adv_mt(BaseRecognizer):

    def __init__(self,
                 backbone,
                 flownet=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 discriminator=None,
                 train_cfg=None,
                 test_cfg=None,
                 seg_head=None,
                 gpus=None,
                 dist=False,
                 train=True):

        super(TSN3D_adv_mt, self).__init__()
        self.tsn3d_backbone = TSN3D_bb_mt(backbone, flownet, spatial_temporal_module, segmental_consensus, cls_head, seg_head, train_cfg, test_cfg)

        if discriminator is not None:
            self.discriminator = builder.build_discriminator(discriminator)

        self.init_weights()

        # put model on gpus
        #pdb.set_trace()
        if train == True:
            if dist == True:
                self.tsn3d_backbone = MMDistributedDataParallel(self.tsn3d_backbone.cuda())
                self.discriminator = MMDistributedDataParallel(self.discriminator.cuda())
            else:
                self.tsn3d_backbone = MMDataParallel(self.tsn3d_backbone, device_ids=range(gpus)).cuda()
                self.discriminator = MMDataParallel(self.discriminator, device_ids=range(gpus)).cuda()
        '''
        assert gpus is not None
        self.backbone = MMDataParallel(self.backbone, device_ids=range(gpus))
        self.spatial_temporal_module = MMDataParallel(self.spatial_temporal_module, device_ids=range(gpus))
        self.segmental_consensus = MMDataParallel(self.segmental_consensus, device_ids=range(gpus))
        self.cls_head = MMDataParallel(self.cls_head, device_ids=range(gpus))
        self.discriminator = MMDataParallel(self.discriminator, device_ids=range(gpus))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        '''

    @property
    def with_discriminator(self):
        return hasattr(self, 'discriminator') and self.discriminator is not None

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label0,
                      gt_label1,
                      **kwargs):
        #assert num_modalities == 1, '%s' % num_modalities
        img_group0 = kwargs['img_group_0']
        seg = kwargs['img_group_seg']
        img_group1 = kwargs['img_group_1']

        # freeze discriminator
        self.discriminator.module.freeze(True)
        feat0, loss_cls0 = self.tsn3d_backbone(img_group0, img_group_seg=seg, gt_label=gt_label0)
        #feat1, loss_cls1 = self.tsn3d_backbone(img_group1, gt_label=gt_label1)
        feat1 = self.tsn3d_backbone(img_group1)

        losses = dict()

        #loss_cls0['loss_cls'].mean().backward()
        #loss_cls0['loss_seg'].mean().backward()
        loss_0 = loss_cls0['loss_cls'] #+ 0.1 * loss_cls0['loss_seg']
        losses['loss_seg'] = loss_cls0['loss_seg']
        loss_0.mean().backward()

        losses['loss_cls0'] = loss_cls0['loss_cls'].mean()
        #losses['loss_cls1'] = loss_cls1['loss_cls'].mean()

        #'''
        outD_1 = self.discriminator(feat1)
        loss_D_1_fake = self.discriminator.module.loss(outD_1, 0)
        #loss_1 = self.discriminator.module.lambda_adv_1 * loss_D_1_fake + loss_cls1['loss_cls']
        loss_1 = self.discriminator.module.lambda_adv_1 * loss_D_1_fake
        loss_1.mean().backward()
        losses['loss_D_1_fake'] = loss_D_1_fake.mean()

        # unfreeze discriminator
        self.discriminator.module.freeze(False)

        feat0, feat1 = feat0.detach(), feat1.detach()
        outD_0 = self.discriminator(feat0)
        loss_D_0_real = self.discriminator.module.loss(outD_0, 0)
        loss_D_0_real.mean().backward()

        outD_1 = self.discriminator(feat1)
        loss_D_1_real = self.discriminator.module.loss(outD_1, 1)
        loss_D_1_real.mean().backward()

        losses['loss_D_0_real'] = loss_D_0_real.mean()
        losses['loss_D_1_real'] = loss_D_1_real.mean()
        #'''

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        #assert num_modalities == 1
        img_group = kwargs['img_group_0']

        return self.tsn3d_backbone(img_group, img_meta, test=True, output_seg=True)
