# optimizer
optimizer_config = dict(grad_clip=None)

optimizer = dict(
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
            }
        )
    )

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5
    )

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(max_keep_ckpts=2, by_epoch=True, interval=1)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU', pre_eval=True)
