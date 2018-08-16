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


class CreditCardBalanceFeaturesOpenSolution(Feature):
    """
    features from https://github.com/neptune-ml/open-solution-home-credit
    """

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        credit_card = CreditCardBalance.get_df(conf)
        static_features = CreditCardBalanceFeaturesOpenSolution._static_features(credit_card)
        dynamic_features = CreditCardBalanceFeaturesOpenSolution._dynamic_features(credit_card)

        features = pd.merge(static_features,
                            dynamic_features,
                            on=['SK_ID_CURR'],
                            validate='one_to_one')
        return features

    @staticmethod
    def _static_features(credit_card):
        credit_card['number_of_installments'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
            'CNT_INSTALMENT_MATURE_CUM']

        credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
            by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        groupby = credit_card.groupby(by=['SK_ID_CURR'])

        g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['SK_DPD'].agg('mean').reset_index()
        g.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['number_of_installments'].agg('sum').reset_index()
        g.rename(index=str, columns={'number_of_installments': 'credit_card_total_installments'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
        g.rename(index=str,
                 columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},
                 inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features[
            'credit_card_drawings_total']

        features['credit_card_installments_per_loan'] = (
            features['credit_card_total_installments'] / features['credit_card_number_of_loans'])

        return features

    @staticmethod
    def _dynamic_features(credit_card):
        features = pd.DataFrame({'SK_ID_CURR': credit_card['SK_ID_CURR'].unique()})

        credit_card_sorted = credit_card.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])

        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])
        credit_card_sorted['credit_card_monthly_diff'] = groupby['AMT_BALANCE'].diff()
        groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])

        g = groupby['credit_card_monthly_diff'].agg('mean').reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        return features
