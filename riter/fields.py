import PIL

"""
Every object is a field which has two parameters.
1) The type of the data itself (e.g string)
2) What is that data when it's indexed
"""
class Field:
    def __init__(self, dtype, indexed):
        self.dtype = dtype
        self.indexed = indexed


class ID(Field):
    """Represent unique ID of a document"""

    def __init__(self):
        super().__init__((str, int, float), False)


class Text(Field):
    """Represent text data.

    Args:
        indexed (bool): ``True`` if this field should be indexed.
    """

    def __init__(self, indexed):
        super().__init__(str, indexed)


class Image(Field):
    """Represent image data

    Args:
        indexed (bool): ``True`` if this field should be indexed.
    """

    def __init__(self, indexed):
        super().__init__(PIL.Image.Image, indexed)


class ImagePath(Field):
    """Represent path of an image. This is used to indicate that during runtime, we should open this file
        and process it as an image.

    Args:
        indexed (bool): ``True`` if this field should be indexed.
    """

    def __init__(self, indexed):
        super().__init__(str, indexed)
