import pandas as pd

from elsa.files import unified_dhodcz
from elsa.root import Elsa

elsa = Elsa.from_unified(files=unified_dhodcz)
truth = elsa.truth
combos = truth.combos
prompts = ~elsa.prompts.isyns.duplicated()

# is_105 = elsa.prompts.isyns == 105
# elsa.prompts.loc[is_105]
# elsa.truth.unique.rephrase.isyn
# elsa.truth.unique.rephrase.loc[105, 'label isyn'.split()]
# # pushing stroller is isyn 12
# loc = elsa.truth.isyns == (7, 12, 23, 27, 30)
# loc = elsa.truth.isyns == (7, 12, 23, 27)
# assert combos.includes(meta='activity').loc[105]
# # pushing stoller is an activity; but the "upgrade" means it is not showing up

# must include state
loc = combos.includes(meta='state')
isyns = combos.isyns.loc[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

# must include activity
loc = combos.includes(meta='activity')
isyns = combos.isyns.loc[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

# 3 of people
# only 1 people combo has state
loc = combos.includes(label='people')
isyns = combos.isyns.loc[loc]
loc = prompts & elsa.prompts.isyns.isin(isyns)
loc &= loc.cumsum() <= 3
people = loc

# 3 of person
loc = combos.includes(label='person')
isyns = combos.isyns.loc[loc]
loc = prompts & elsa.prompts.isyns.isin(isyns)
loc &= ~people
loc &= loc.cumsum() <= 3
person = loc

# 3 of couple
loc = combos.includes(label='couple')
isyns = combos.isyns.loc[loc]
loc = prompts & elsa.prompts.isyns.isin(isyns)
loc &= ~people
loc &= ~person
loc &= loc.cumsum() <= 3
couple = loc

prompts = people | person | couple

# select 20 images which contain at least one of the prompts
isyns = elsa.prompts.isyns.loc[prompts]
loc = combos.isyns.isin(isyns)
file = elsa.truth.combos.file.loc[loc]
files = elsa.files.file.isin(file)
files &= files.cumsum() <= 20

# # todo: get first for each unique isyn w/ needles, haystack
# pd.set_option('display.max_colwidth', None)
elsa.files.file.loc[files]
elsa.prompts.natural.loc[prompts]
# # elsa.prompts.reset_index('isyns').loc[1332]
# elsa.truth.unique.rephrase.meta
# loc = elsa.truth.unique.rephrase.irephrase == 1344
# elsa.truth.unique.rephrase.loc[loc, 'label meta'.split()]
#
# elsa.truth.unique.rephrase.includes(label='person')
#
# loc = combos.includes('gathering')
# loc &= combos.includes('crossing road')
# loc &= combos.includes('walking')
# loc &= combos.includes('people')
# loc &= combos.includes('child')
# ibox = combos.ibox[loc]
# ibox = [764, 923, 3363, 3497, 4109]
# # loc = elsa.truth.ibox.isin(ibox)
# loc = elsa.truth.ibox == 764
# elsa.truth.isyns.loc[loc]
# (27, 23,12,7)
#
# tuples = elsa.truth.combos.isyns.unique().tolist()
# tuples = elsa.truth.isyns.unique().tolist()
# tuples = elsa.prompts.isyns.unique().tolist()
# for t in tuples:
#     assert all(t[i] <= t[i + 1] for i in range(len(t) - 1)), f"Tuple {t} is not sorted in ascending order"


# elsa.truth.unique.rephrase.includes(label='people').loc

"""
Out[6]: Index([764, 923, 3363, 3497, 4109], dtype='int64', name='ibox')

somehow label=people has been dropped?

truth.unique.rephrase
                           label       meta
isyns irephrase                          
104     1344           gathering  condition
        1344       crossing road   activity
        1344             walking      state

"""