# dataset config
_base_ = [
    "../_base_/datasets/cityscapes_1024x1024.py",
    "../_base_/default_runtime.py",
    "../_base_/models/rein_dinov2_mask2former.py",
]
crop_size = (1024, 1024)
num_classes = 19
model = dict(
    backbone=dict(
        img_size=1024,
        init_cfg=dict(
            checkpoint="checkpoints/dinov2_converted_1024x1024_p16.pth",
        ),
    ),
    decode_head=dict(
        type="ReinMask2FormerHead2",
        loss_con=dict(ood_idx=num_classes, temperature=0.10, loss_weight=2.0),
        num_classes=num_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])
    ),
    data_preprocessor=dict(
        size=crop_size,
    ),
    test_cfg=dict(
        crop_size=crop_size,
        stride=(683, 683),
    ),
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="MixCOCO", coco_root="/home/johnl/data/coco/"),   #* Change this to your own path
    dict(
        type="RandomChoiceResize",
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=4096,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4, 
    dataset=dict(type="CityscapesCOCODataset", pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.00006, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=10000),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=0.9,
        begin=10000,
        end=80000,
        by_epoch=False,
    ),
]
# training schedule for 160k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=8000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
find_unused_parameters = True
