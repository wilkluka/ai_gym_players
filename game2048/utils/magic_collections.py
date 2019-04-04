from collections import UserList
import numpy as np


class MagicList(UserList):

    @staticmethod
    def ret_magic_list(func):
        def inner(*args, **kwargs):
            return MagicList(func(*args, **kwargs))
        return inner

    @ret_magic_list.__get__(object)
    def flatten(self):
        def flatten_list(seq):
            if isinstance(seq, list):
                result = []
                for elem in seq:
                    if isinstance(elem, list):
                        result.extend(flatten_list(elem))
                    else:
                        result.append(elem)
                return result
            else:
                return [seq]
        res = []
        for elem in self:
            res.extend(flatten_list(elem))
        return res

    def to_list(self):
        return list(self)

    def to_ndarray(self, flatten=True):
        if flatten:
            lst = self.flatten()
        else:
            lst = list(self)
        return np.array(lst)

    @ret_magic_list.__get__(object)
    def map(self, mapper=lambda x: x, filter_func=lambda x: True):
        return [mapper(elem) for elem in self if filter_func(elem)]

    @ret_magic_list.__get__(object)
    def split(self, indexes):
        cut = [elem for i, elem in enumerate(self) if i in indexes]
        remains = [elem for i, elem in enumerate(self) if i not in indexes]
        return remains, cut


MList = MagicList
