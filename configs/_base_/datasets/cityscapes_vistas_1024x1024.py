

cityscapes_root = "/home/johnl/data/cityscapes/"    #* Change this to your own path
vistas_root = "/home/johnl/data/vistas/"            #* Change this to your own path

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type='MapillaryHack'),
    dict(type="Resize", scale=(2048, 1024)),
    dict(type="RandomCrop", crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_cityscapes = dict(
    type="CityscapesDataset",
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/train",
        seg_map_path="gtFine/train",
    ),
    pipeline=train_pipeline,
)
train_vistas = dict(
    type="MapillaryDataset",
    data_root=vistas_root,
    data_prefix=dict(
        img_path="training/images",
        seg_map_path="training/v1.2/labels"
    ),
    pipeline=train_pipeline,
)
val_cityscapes = dict(
    type="CityscapesDataset",
    data_root=cityscapes_root,
    data_prefix=dict(
        img_path="leftImg8bit/val",
        seg_map_path="gtFine/val",
    ),
    pipeline=test_pipeline,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            train_cityscapes,
            train_vistas,
        ],
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_cityscapes,
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', 
                     iou_metrics=['mIoU'])
test_evaluator = val_evaluator