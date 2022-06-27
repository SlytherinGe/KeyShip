# evaluation
evaluation = dict(interval=1, metric='details', save_best='auto')
# optimizer
optimizer = dict(type='Adam', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step',
                warmup='linear',
                warmup_iters=50,
                warmup_ratio=1.0 / 3, 
                step=[150, 190])
runner = dict(type='EpochBasedRunner', max_epochs=210)
checkpoint_config = dict(interval=10)