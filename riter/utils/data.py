from collections import OrderedDict
from typing import Union, List
from PIL import Image

import torch

from riter.utils import logger
from riter.fields import ImagePath


class Document(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(OrderedDict, self).__init__(*args, **kwargs)
        if "id" not in self:
            raise ValueError(f"Document must contain key")
        try:
            hash(self["id"])
        except TypeError:
            raise TypeError(f"`id` of a document must be hashable.")

    def __eq__(self, value):
        if not isinstance(value, Document):
            raise TypeError(
                f"Cannot directly compare document with object of type '{type(value)}'."
            )
        return self["id"] == value["id"]

    def __hash__(self):
        return self["id"]

    @property
    def id(self):
        return self["id"]


class DocDataset(torch.utils.data.Dataset):
    """Internal dataset class for processing documents"""

    def __init__(self, documents: List[Document], schema):
        self.documents = documents
        self.schema = schema

    def set_field(self, field: str):
        self.field = field

    def __getitem__(self, index):
        doc = self.documents[index]
        if isinstance(self.schema[self.field], ImagePath):
            return Image.open(doc[self.field]).convert("RGB")
        else:
            return doc[self.field]

    def __len__(self):
        return len(self.documents)
