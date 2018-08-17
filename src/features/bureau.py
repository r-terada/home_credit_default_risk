import os
import gc
import numpy as np
import pandas as pd

from features import one_hot_encoder, Feature
from features.raw_data import Bureau, BureauBalance


class BureauFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        bureau = Bureau.get_df(conf)
        bureau, bureau_cat = one_hot_encoder(bureau, True)

        bb = BureauBalance.get_df(conf)
        bb, bb_cat = one_hot_encoder(bb, True)

        # Bureau balance: Perform aggregations and merge with bureau.csv
        bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in bb_cat:
            bb_aggregations[col] = ['mean']
        bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
        bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
        bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        del bb, bb_agg
        gc.collect()

        # Bureau and bureau_balance numeric features
        num_aggregations = {
            'DAYS_CREDIT': ['mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM': ['mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'MONTHS_BALANCE_MIN': ['min'],
            'MONTHS_BALANCE_MAX': ['max'],
            'MONTHS_BALANCE_SIZE': ['mean', 'sum']
        }
        # Bureau and bureau_balance categorical features
        cat_aggregations = {}
        for cat in bureau_cat:
            cat_aggregations[cat] = ['mean']
        for cat in bb_cat:
            cat_aggregations[cat + "_MEAN"] = ['mean']

        bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
        # Bureau: Active credits - using only numerical aggregations
        active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
        active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
        del active, active_agg
        gc.collect()
        # Bureau: Closed credits - using only numerical aggregations
        closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
        closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
        del closed, closed_agg, bureau
        gc.collect()
        return bureau_agg


class BureauFeaturesOpenSolution(Feature):
    """
    features from https://github.com/neptune-ml/open-solution-home-credit
    """

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        bureau = Bureau.get_df(conf)
        bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
        bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        features = pd.DataFrame({'SK_ID_CURR': bureau['SK_ID_CURR'].unique()})

        groupby = bureau.groupby(by=['SK_ID_CURR'])

        g = groupby['DAYS_CREDIT'].agg('count').reset_index()
        g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
        g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_active_binary': 'bureau_credit_active_binary'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
        g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        features['bureau_average_of_past_loans_per_type'] = \
            features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

        features['bureau_debt_credit_ratio'] = \
            features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

        features['bureau_overdue_debt_ratio'] = \
            features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

        return features
