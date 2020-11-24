from collections import OrderedDict


class Document:
    def __init__(self, guid, data):
        self.guid = guid
        self._data = OrderedDict(data)

    @property
    def fields(self):
        return self._data.keys()

    def __getitem__(self, key):
        if key == "guid":
            return self.guid
        else:
            return self._data[key]