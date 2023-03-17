from easydict import EasyDict as edict

cfg = edict({
    "device_target": "GPU",
    "dataset_sink_mode": False,
    "model_name": "unet",
    "output_path": './output',
    "checkpoint_path": "./checkpoint/",
    "checkpoint_file_path": "ckpt_unet_adam_2-200_91.ckpt",
    "keep_checkpoint_max": 10,
    "per_print_times": 0,
    
    "dataset": "VOC2012",
    "ignore_label": 255,
    "num_cls": 21,
    "repeat": 1,

    "train_batch_size": 16,
    "train_crop_size": 128,
    "train_image_mean": [103.53, 116.28, 123.675],
    "train_image_std": [57.375, 57.120, 58.395],
    "train_min_scale": 0.5,
    "train_max_scale": 2.0,
    "train_repeat": 1,

    "test_batch_size": 1,
    "test_crop_size": 256,
    "test_image_mean": [103.53, 116.28, 123.675],
    "test_image_std": [57.375, 57.120, 58.395],
    "test_min_scale": 1,
    "test_max_scale": 1,

    "epochs": 200,
    "lr": 0.00001,
    "weight_decay": 0,
    "loss_scale": 1024.0,
    "FixedLossScaleManager": 1024.0
})