# model settings
model = dict(
    type='TSN3D_adv_mt',
    backbone=dict(
        type='ResNet_I3D',
        pretrained='modelzoo://resnet50',
        depth=50,
        num_stages=4,
        out_indices=[0,1,2,3],
        frozen_stages=-1,
        inflate_freq=((1,1,1), (1,0,1,0), (1,0,1,0,1,0), (0,1,0)),
        inflate_style='3x1x1',
        conv1_kernel_t=5,
        conv1_stride_t=2,
        pool1_kernel_t=1,
        pool1_stride_t=2,
        bn_eval=False,
        partial_bn=False,
        style='pytorch'),
    spatial_temporal_module=dict(
        type='SimpleSpatialTemporalModule',
        spatial_type='avg',
        temporal_size=4,
        spatial_size=7),
    segmental_consensus=dict(
        type='SimpleConsensus',
        consensus_type='avg'),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_feature_size=1,
        spatial_feature_size=1,
        dropout_ratio=0.5,
        in_channels=2048,
        num_classes=400),
    seg_head=dict(
        type='SegHead',
        n_classes=2,
        input_size=224
        ),
    discriminator=dict(
        type='NLayerDiscriminator',
        input_nc=2048,
        lambda_adv_1=0.001
        ))
train_cfg = None
test_cfg = None
# dataset settings
dataset_type = 'RawFramesDatasetAdv'
dataset_type_eval = 'RawFramesDataset'
data_root0 = 'data/unreal/rawframes_train/'
data_root1 = 'data/nturgbd/rawframes_train/'
data_root_val = 'data/nturgbd/rawframes_val/'
data_root_test = 'data/nturgbd/rawframes_val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file0='data/unreal/unreal_train_split_rawframes.txt',
        ann_file1='data/nturgbd/nturgbd_train_split_cross_setup_rawframes.txt',
        img_prefix0=data_root0,
        img_prefix1=data_root1,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=32,
        new_step=2,
        random_shift=True,
        modality=['RGB','Seg'],
        image_tmpl0=['img_{:08d}.jpg', 'seg_{:08d}.png'],
        image_tmpl1='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0.5,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=True,
        scales=[1, 0.8],
        max_distort=0,
        test_mode=False),
    val=dict(
        type=dataset_type_eval,
        ann_file='data/nturgbd/nturgbd_val_split_cross_setup_rawframes_partial.txt',
        img_prefix=data_root_val,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=1,
        new_length=32,
        new_step=2,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=224,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample=None,
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=False),
    test=dict(
        type=dataset_type_eval,
        ann_file='data/nturgbd/nturgbd_val_split_cross_setup_rawframes.txt',
        #ann_file='data/kinetics400/kinetics400_val_list_rawframes_ntu.txt',
        img_prefix=data_root_test,
        img_norm_cfg=img_norm_cfg,
        input_format="NCTHW",
        num_segments=3,
        new_length=32,
        new_step=2,
        random_shift=True,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=256,
        div_255=False,
        flip_ratio=0,
        resize_keep_ratio=True,
        oversample='three_crop',
        random_crop=False,
        more_fix_crop=False,
        multiscale_crop=False,
        test_mode=True))
# optimizer
optimizers = [  dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
                dict(type='Adam', lr=0.0001, betas=(0.9, 0.99)),
        ]
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_configs = [
        dict(policy='step',
             step=[30, 60]),
        dict(policy='step',
             step=[30, 60]),
    ]
checkpoint_config = dict(interval=1)
# workflow = [('train', 5), ('val', 1)]
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/i3d_SimNTU_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2_b8_g8_adv_4680_mt'
load_from = None
resume_from = None



