import pandas as pd
pd.options.mode.chained_assignment = 'raise'
import warnings
warnings.filterwarnings('error', category=pd.errors.PerformanceWarning)

import warnings
warnings.filterwarnings('error', message='Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas')


import warnings

warnings.filterwarnings('error', category=FutureWarning, message='.*default of observed=False.*')
pd.errors.SettingWithCopyWarning
warnings.filterwarnings('error', category=pd.errors.SettingWithCopyWarning, message='.*A value is trying to be set on a copy.*')
import pandas as pd

pd.options.mode.chained_assignment = 'raise'

warnings.filterwarnings('error', category=RuntimeWarning, message='.*scalar divide.*')