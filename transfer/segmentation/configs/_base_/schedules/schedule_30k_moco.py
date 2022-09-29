# optimizer
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[21000, 27000], gamma=0.1, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=3000)
evaluation = dict(interval=3000, metric='mIoU', pre_eval=True)
