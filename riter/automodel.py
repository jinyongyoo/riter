from collections import OrderedDict
from .zoo.vsepp import VseppImageEncoder, VseppTextEncoder

MODEL_CLS_MAPPING = OrderedDict(
    [
        ("vsepp-resnet-coco", VseppImageEncoder),
        ("vsepp-vgg-coco", VseppImageEncoder),
        ("vsepp-gru-resnet-coco", VseppTextEncoder),
        ("vsepp-gru-vgg-coco", VseppTextEncoder),
        ("vsepp-resnet-flickr", VseppImageEncoder),
        ("vsepp-vgg-flickr", VseppImageEncoder),
        ("vsepp-gru-resnet-flickr", VseppTextEncoder),
        ("vsepp-gru-vgg-flickr", VseppTextEncoder),
    ]
)


class AutoModel:
    r"""
    This is a generic model class that will be instantiated as one of the base model classes of the library when
    created with the :meth:`~riter.AutoModel.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(prebuilt_model_name)`."
        )

    @classmethod
    def from_pretrained(cls, prebuilt_model_name: str):
        r"""

        Examples::

            >>> from riter import AutoModel

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoModel.from_pretrained('vsepp-resnet-coco')
        """
        if prebuilt_model_name not in MODEL_CLS_MAPPING:
            raise ValueError(
                f"`{prebuilt_model_name}` is not an available pretrained model."
            )

        model_cls = MODEL_CLS_MAPPING[prebuilt_model_name]
        model = model_cls.from_pretrained(prebuilt_model_name)

        return model
