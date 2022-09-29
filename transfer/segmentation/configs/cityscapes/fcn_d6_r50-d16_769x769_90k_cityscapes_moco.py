_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes_769x769_moco.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_90k_moco.py'
]
data = dict(
    samples_per_gpu=8,  # remember to modify it when changing gpu number
    workers_per_gpu=4
)
model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1)),
    decode_head=dict(channels=256, concat_input=False,
                     dropout_ratio=0, align_corners=True, dilation=6),
    auxiliary_head=None,
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
