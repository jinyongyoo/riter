import torchvision
from collections import OrderedDict

VSEPP_TRANSFORMATIONS = OrderedDict(
    [
        (
            "vsepp-resnet-coco",
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        (
            "vsepp-vgg-coco",
            torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
    ]
)


class VseppTransformation:
    def __init__(self, transformation):
        self.transformation = transformation

    def __call__(self, img):
        return self.transformation(img)

    @classmethod
    def from_pretrained(cls, pretrained_name):
        if pretrained_name not in VSEPP_TRANSFORMATIONS:
            raise ValueError(
                f"`{pretrained_name}` is not available. "
                f"Available options are: {list(VSEPP_TRANSFORMATIONS.keys())}"
            )
        return VseppTransformation(VSEPP_TRANSFORMATIONS[pretrained_name])
