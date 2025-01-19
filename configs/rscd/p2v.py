_base_ = [
    '../_base_/datasets/rscd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

data_root = '/home/ps/HDD/zhaoyq_data/CDdata/CD_Data_GZ/cut_data'
crop_size = (256, 256)

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
find_unused_parameters=True
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='MM_P2V',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='NullDecodeHead',
        in_index=0,
        in_channels=256,
        channels=512,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='NullDecodeHead',
        in_index=1,
        in_channels=256,
        channels=512,
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    # dict(type='CLAHE'),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomRotate', prob=0.5, degree=30),
    # dict(type='AdjustGamma'),
    dict(type='PhotoMetricDistortion'),
    # dict(type='RandomRotFlip'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]


train_dataloader = dict(batch_size=16,
                        num_workers=16,
                        dataset=dict(data_root=data_root,
                                    pipeline=train_pipeline))
val_dataloader = dict(batch_size=1,
                        dataset=dict(data_root=data_root))
test_dataloader = dict(batch_size=1,
                        dataset=dict(data_root=data_root))

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))