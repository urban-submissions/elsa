from __future__ import annotations

import copy
from collections import UserDict

import pandas.core.generic
from pandas.core.groupby import DataFrameGroupBy

"""
https://github.com/pandas-dev/pandas/pull/55314
ENH: propagating attrs always uses deepcopy #55314

I am not a fan of this pull request. 
__finalize__ calls the following line:
`self.attrs = deepcopy(other.attrs)`
__finalize__ is called fairly frequently: 
for example. Frame.loc[...] calls __finalize__;
pretty much everything calls finalize. We are making
huge amounts of calls to deepcopy, and in my library, 
other DataFrames are stored in attrs. This is a huge
performance hit.
"""

if False:
    from .ndframe import NDFrame


class Attrs(UserDict):
    # todo: likely delete this module;
    # Attrs is not JSON serializable; causes issues with NDFrame.to_parquet

    def __eq__(self, other):
        # return false such that concatenate does not try to call
        # DataFrame.__eq__ on magic.Frame, which will fail
        return False

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return instance.__dict__['attrs']
        except KeyError:
            instance.__dict__['attrs'] = self.__class__()
        return instance.__dict__['attrs']

    def __set__(self, instance: NDFrame, value: dict):
        instance.__dict__['attrs'] = self.__class__(value)

    def __delete__(self, instance):
        try:
            del instance.__dict__['attrs']
        except KeyError:
            pass

    def __deepcopy__(self, memodict=None):
        return copy.copy(self)

    # todo: maybe pandas.attrs could have _deepcopy=False or True


if __name__ == '__main__':
    att = Attrs({
        'a': 'arst',
        'b': 'arstarstarstoarst'
    })
    cp = copy.deepcopy(att)
    assert cp['a'] is att['a']
    assert cp['b'] is att['b']

pandas.core.generic.NDFrame.__finalize__
