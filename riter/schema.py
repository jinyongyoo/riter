from collections import OrderedDict

from riter import utils
from .document import Document

class Schema:
    def __init__(self, schema=OrderedDict()):
        self._schema = schema

    def add_field(self, name: str, data_type, index_type):
        if name in self._schema:
            logger.info(f"Warning: field with name '{name}' already exists in the schema.")

        self._schema[name] = (data_type, index_type)

    def remove_field(self, name: str):
        if name in self._schema:
            del self._schema[name]

    def validate_document(self, doc: Document) -> bool:
        for key, value in self._schema.item():
            if key not in doc.fields:
                return False
            else:
                dtype, _ = value
                if isinstance(self._schema[key], dtype):
                    return True
                else:
                    return False


