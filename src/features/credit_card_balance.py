import gc
import numpy as np
import pandas as pd

from features import one_hot_encoder, Feature
from features.raw_data import CreditCardBalance


class CreditCardBalanceFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        cc = CreditCardBalance.get_df(conf)
        cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
        cc.loc[cc['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan
        cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
        # General aggregations
        cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
        cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
        cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
        # Count credit card lines
        cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
        del cc
        gc.collect()
        return cc_agg
