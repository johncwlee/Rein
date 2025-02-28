allo_type = "ALLODataset"
allo_root = "/home/johnl/data/allo/"    #**NOTE: Change this to your own path
allo_crop_size = (1080, 1080)
allo_image_size = (1920, 1080)
allo_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=allo_image_size),
    dict(type="RandomCrop", crop_size=allo_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
allo_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=allo_image_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_allo = dict(
    type=allo_type,
    seed=42,
    data_root=allo_root,
    data_prefix=dict(path="train_v3"),
    pipeline=allo_train_pipeline,
)
val_allo = dict(
    type=allo_type,
    seed=42,
    data_root=allo_root,
    data_prefix=dict(path="test_v3"),
    pipeline=allo_test_pipeline,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=train_allo,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_allo,
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', 
                     iou_metrics=['mIoU'])
test_evaluator = val_evaluator