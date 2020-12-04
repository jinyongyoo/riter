from collections import OrderedDict
from PIL.Image import Image

from riter import utils
from .fields import Field


class Schema(OrderedDict):
    """
    Schema exists because our dataset may not contain just images. There are 
    other important factors we may care about (e.g captions, image titles, etc..)
    """
    
    def __init__(self, *args, **kwargs):
        super(OrderedDict, self).__init__(*args, **kwargs)

    def add_field(self, name: str, field: Field):
        if name in self:
            utils.logger.info(
                f"Warning: field with name '{name}' already exists in the data schema."
            )
        self[name] = field

    def check(self, doc: utils.data.Document) -> bool:
        for name, field in self.items():
            if key not in doc.fields:
                return False
            else:
                return utils.type_check(doc[key], field.dtype)



class IndexRecipe(OrderedDict):
    """
    The data of every field has to be indexed. We have faiss indexing for 
    images and gensim indexing for texts.
    """

    def __init__(self, *args, **kwargs):
        super(OrderedDict, self).__init__(*args, **kwargs)

    def add_faiss_index_recipe(self, name, ndim):
        if name in self:
            utils.logger.info(
                f"Warning: field with name '{name}' already exists in the index schema."
            )

        self[name] = {"type": "faiss", "ndim": ndim}

    def add_gensim_index_recipe(self, name):
        if name in self:
            utils.logger.info(
                f"Warning: field with name '{name}' already exists in the index schema."
            )
        self[name] = {"type": "gensim"}

    def indexable_fields(self):
        return self.keys()
