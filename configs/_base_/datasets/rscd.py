# dataset settings
dataset_type = 'BaseTXTCDDataset'
data_root = 'data/LEVIRCD'
metainfo = dict(
        classes=('background', 'changed'),
        palette=[[0, 0, 0], [255, 255, 255]],
        label_map={0: 0, 255: 1})

albu_train_transforms = [
    dict(type='RandomBrightnessContrast', p=0.2),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5)
]

train_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Albu', transforms=albu_train_transforms),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ConcatCDInput'),
    dict(type='PackSegInputs')
]

tta_pipeline = [
    dict(type='LoadMultipleRSImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[dict(type='LoadAnnotations')],
                    [dict(type='ConcatCDInput')],
                    [dict(type='PackSegInputs')]])
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/A',
            img_path2='train/B',
            seg_map_path='train/label'),
        ann_file='train.txt',
        metainfo=metainfo,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        ann_file='val.txt',
        metainfo=metainfo,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/A', img_path2='test/B', seg_map_path='test/label'),
        ann_file='test.txt',
        metainfo=metainfo,
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
