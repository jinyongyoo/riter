COCO_RESNET_CONFIG = {
    "embed_size": 1024,
    "finetune": False,
    "cnn_type": "resnet152",
    "use_abs": False,
    "no_imgnorm": False,
}

COCO_VGG_CONFIG = {
    "embed_size": 1024,
    "finetune": False,
    "cnn_type": "vgg19",
    "use_abs": False,
    "no_imgnorm": False,
}

COCO_GRU_CONFIG = {
    "vocab_size": 11755,
    "word_dim": 300,
    "embed_size": 1024,
    "num_layers": 1,
    "use_abs": False,
}
