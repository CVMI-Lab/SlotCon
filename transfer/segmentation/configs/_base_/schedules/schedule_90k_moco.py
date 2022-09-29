# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='step', step=[63000, 81000], gamma=0.1, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=9000)
evaluation = dict(interval=9000, metric='mIoU', pre_eval=True)
