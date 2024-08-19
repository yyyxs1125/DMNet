_base_ = [
    '../_base_/models/dmnet_r50-d8.py', '../_base_/datasets/my_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
NUM_CLASSES = 2
model = dict(
    decode_head=dict(num_classes=NUM_CLASSES),  
    auxiliary_head=dict(num_classes=NUM_CLASSES)
)