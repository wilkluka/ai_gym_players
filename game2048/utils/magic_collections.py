from collections import UserList
import numpy as np


class MagicList(UserList):

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

    def extract(self, extractor):
        return [extractor(elem) for elem in self]
