import flax
from collections import OrderedDict
from argparse import Namespace


def namespace_to_dict(ns, copy=True):
    d = vars(ns)
    if copy:
        d = d.copy()
    return d


class Config(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def as_dict(self):
        return namespace_to_dict(self)

    def __getitem__(self, key):
        return self.__getattribute__(key)
