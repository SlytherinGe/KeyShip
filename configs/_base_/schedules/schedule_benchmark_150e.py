evaluation = dict(interval=1, metric='details', save_best='auto')
# optimizer
optimizer = dict(type='Adam', lr=0.0006)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=150)
lr_config = dict(policy='step',
                warmup='linear',
                warmup_iters=50,
                warmup_ratio=1.0 / 3,
                 step=[110])
checkpoint_config = dict(interval=150)