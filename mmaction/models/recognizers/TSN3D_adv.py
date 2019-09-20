from .base import BaseRecognizer
from .. import builder
from ..registry import RECOGNIZERS

import torch


@RECOGNIZERS.register_module
class TSN3D_adv(BaseRecognizer):

    def __init__(self,
                 backbone,
                 flownet=None,
                 spatial_temporal_module=None,
                 segmental_consensus=None,
                 cls_head=None,
                 discriminator=None
                 train_cfg=None,
                 test_cfg=None):

        super(TSN3D, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if flownet is not None:
            self.flownet = builder.build_flownet(flownet)

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            raise NotImplementedError

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            raise NotImplementedError

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            raise NotImplementedError

        if discriminator is not None:
            self.discriminator = builder.build_discriminator(discriminator)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()

    @property
    def with_flownet(self):
        return hasattr(self, 'flownet') and self.flownet is not None

    @property
    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    @property
    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    @propoerty
    def with_discriminator(self):
        return hasattr(self, 'discriminator') and self.discriminator is not None

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        super(TSN3D, self).init_weights()
        self.backbone.init_weights()

        if self.with_flownet:
            self.flownet.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

    def extract_feat_with_flow(self, img_group,
                               trajectory_forward=None,
                               trajectory_backward=None):
        x = self.backbone(img_group,
                          trajectory_forward=trajectory_forward,
                          trajectory_backward=trajectory_backward)
        return x

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      gt_label0,
                      gt_label1,
                      **kwargs):
        assert num_modalities == 1
        img_group0 = kwargs['img_group_0']
        img_group1 = kwargs['img_group_1']

        bs = img_group0.shape[0]
        img_group0 = img_group0.reshape((-1, ) + img_group0.shape[2:])
        num_seg = img_group0.shape[0] // bs

        # freeze discriminator
        self.discriminator.freeze(True)
        if self.with_flownet:
            raise NotImplementedError
        else:
            feat0 = self.extract_feat(img_group0)
            feat1 = self.extract_feat(img_group1)
        if self.with_spatial_temporal_module:
            feat0 = self.spatial_temporal_module(feat0)
            feat1 = self.spatial_temporal_module(feat1)
        if self.with_segmental_consensus:
            feat0 = feat0.reshape((-1, num_seg) + feat0.shape[1:])
            feat0 = self.segmental_consensus(feat0)
            feat0 = feat0.squeeze(1)
            feat1 = feat1.reshape((-1, num_seg) + feat1.shape[1:])
            feat1 = self.segmental_consensus(feat1)
            feat1 = feat1.squeeze(1)
        losses = dict()
        if self.with_flownet:
            raise NotImplementedError

        if self.with_cls_head:
            cls_score = self.cls_head(feat0)
            gt_label0 = gt_label0.squeeze()
            loss_cls0 = self.cls_head.loss(cls_score0, gt_label0)
            loss_cls0.backward()

            cls_score = self.cls_head(feat1)
            gt_label1 = gt_label1.squeeze()
            loss_cls1 = self.cls_head.loss(cls_score1, gt_label1)


            losses['loss_cls0'] = loss_cls0
            losses['loss_cls1'] = loss_cls1

            outD_1 = self.discriminator(feat1)
            loss_D_1_fake = self.discriminator.loss(outD_1, 0)
            loss_1 = lambda_adv_1 * loss_D_1_fake + loss_cls1
            loss_1.backward()
            losses['loss_D_1_fake'] = loss_D_1_fake

            # unfreeze discriminator
            self.discriminator.freeze(False)

            feat0, feat1 = feat0.detach(), feat1.detach()
            outD_0 = self.discriminator(feat0)
            loss_D_0_real = self.discriminator.loss(outD_0, 0)
            loss_D_0_real.backward()

            outD_1 = self.discriminator(feat1)
            loss_D_1_real = self.discriminator.loss(outD_1, 1)
            loss_D_1_real.backward()

            losses['loss_D_0_real'] = loss_D_0_real
            losses['loss_D_1_real'] = loss_D_1_real


        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        bs = img_group.shape[0]
        img_group = img_group.reshape((-1, ) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs

        if self.with_flownet:
            raise NotImplementedError
        else:
            x = self.extract_feat(img_group)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = x.reshape((-1, num_seg) + x.shape[1:])
            x = self.segmental_consensus(x)
            x = x.squeeze(1)
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy()
