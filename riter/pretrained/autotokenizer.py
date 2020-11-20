from collections import OrderedDict
from .vsepp import VseppTokenizer

TOKENIZER_CLS_MAPPING = OrderedDict(
    [
        ("vsepp-gru-coco", VseppTokenizer),
    ]
)


class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the base model classes of the library when
    created with the :meth:`~riter.AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTokenizer is designed to be instantiated "
            "using the `AutoTokenizer.from_pretrained(prebuilt_model_name)`."
        )

    @classmethod
    def from_pretrained(cls, prebuilt_model_name):
        r"""

        Examples::

            >>> from riter import AutoTokenizer

            >>> # Download model and configuration from S3 and cache.
            >>> model = AutoTokenizer.from_pretrained('vsepp-gru-coco')
        """
        if prebuilt_model_name not in TOKENIZER_CLS_MAPPING:
            raise ValueError(
                f"`{prebuilt_model_name}` is not an available pretrained tokenizer."
            )

        tokenizer_cls = TOKENIZER_CLS_MAPPING[prebuilt_model_name]
        tokenizer = tokenizer_cls.from_pretrained(prebuilt_model_name)

        return tokenizer
