from collections import OrderedDict
from .vsepp import VseppTransformation

TRANSFORMATION_CLS_MAPPING = OrderedDict(
    [
        ("vsepp-resnet-coco", VseppTransformation),
        ("vsepp-vgg-coco", VseppTransformation),
    ]
)


class AutoTransformation:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the base model classes of the library when
    created with the :meth:`~riter.AutoTransformation.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTransformation is designed to be instantiated "
            "using the `AutoTransformation.from_pretrained(prebuilt_model_name)`."
        )

    @classmethod
    def from_pretrained(cls, prebuilt_model_name):
        r"""

        Examples::

            >>> from riter import AutoTransformation

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoTransformation.from_pretrained('vsepp-gru-coco')
        """
        if prebuilt_model_name not in TRANSFORMATION_CLS_MAPPING:
            raise ValueError(
                f"`{prebuilt_model_name}` is not an available pretrained tokenizer."
            )

        transformation_cls = TRANSFORMATION_CLS_MAPPING[prebuilt_model_name]
        transformation = transformation_cls.from_pretrained(prebuilt_model_name)

        return transformation
