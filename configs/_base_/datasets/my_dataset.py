dataset_type = 'MyDataset'
data_root = 'data/my_dataset'

# mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
img_norm_cfg = dict(
     mean=[85.6545, 96.237, 73.3335], std=[43.942, 44.778, 44.829], to_rgb=True)
crop_size = (640, 400)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(1920, 1200),ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',  # 使用调整图像大小(resize)增强
        scale=crop_size,  # 图像缩放的大小
        keep_ratio=True),  # 在调整图像大小时是否保留长宽比
    dict(type='LoadAnnotations'),  # 加载数据集提供的语义分割标注
    dict(type='PackSegInputs') 
]

train_dataloader = dict(  # 训练数据加载器(dataloader)的配置
    batch_size=28,  # 每一个GPU的batch size大小
    num_workers=8,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 训练时进行随机洗牌(shuffle)
    dataset=dict(  # 训练数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='split_image/train', seg_map_path='split_anno/train'),  # 训练数据的前缀
        pipeline=train_pipeline)) # 数据处理流程，它通过之前创建的train_pipeline传递。

val_dataloader = dict(
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='split_image/val', seg_map_path='split_anno/val'),  # 测试数据的前缀
        pipeline=test_pipeline))  # 数据处理流程，它通过之前创建的test_pipeline传递。

test_dataloader = dict(
    batch_size=1,  # 每一个GPU的batch size大小
    num_workers=4,  # 为每一个GPU预读取数据的进程个数
    persistent_workers=True,  # 在一个epoch结束后关闭worker进程，可以加快训练速度
    sampler=dict(type='DefaultSampler', shuffle=False),  # 训练时不进行随机洗牌(shuffle)
    dataset=dict(  # 测试数据集配置
        type=dataset_type,  # 数据集类型，详见mmseg/datassets/
        data_root=data_root,  # 数据集的根目录
        data_prefix=dict(
            img_path='split_image/test', seg_map_path='split_anno/test'),  # 测试数据的前缀
        pipeline=test_pipeline))

# 精度评估方法，我们在这里使用 IoUMetric 进行评估
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
