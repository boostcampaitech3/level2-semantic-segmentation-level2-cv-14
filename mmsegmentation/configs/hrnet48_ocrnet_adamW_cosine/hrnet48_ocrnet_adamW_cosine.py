_base_ = [
    'dataset.py',
    'schedule.py',
    'runtime.py',
    'models.py'
]
# 총 Epoch size
runner = dict(max_epochs=100)

# samples_per_gpu -> batch size라 생각하면 됨
data = dict(samples_per_gpu=12)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
        init_kwargs=dict(
            # 각각 자신에 맞춰서 Project이름 설정
            project= 'Semantic_Segmentation',
            name = '[Taeha2]HRNet_48_OCRNet_adamW_cosine(ADE20K)',
            tags = 'T',
            ),
            # log_checkpoint=True,
            # log_checkpoint_metadata=True,
            # num_eval_images=10
        ),
        # dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(max_keep_ckpts=2, by_epoch=True, interval=1)
seed=42