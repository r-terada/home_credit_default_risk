import gc
import numpy as np
import pandas as pd
from functools import partial

from category_encoders import TargetEncoder
from features import Feature, one_hot_encoder, parallel_apply, add_features_in_group, add_trend_feature
from features.base import Base
from features.raw_data import PosCash


class PosCashFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        pos = PosCash.get_df(conf)
        pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
        # Features
        aggregations = {
            'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']

        pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
        pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
        # Count pos cash accounts
        pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
        del pos
        gc.collect()
        return pos_agg


class PosCashFeaturesOpenSolution(Feature):
    """
    features from https://github.com/neptune-ml/open-solution-home-credit
    """

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        pos_cash = PosCash.get_df(conf)
        pos_cash['is_contract_status_completed'] = pos_cash['NAME_CONTRACT_STATUS'] == 'Completed'
        pos_cash['pos_cash_paid_late'] = (pos_cash['SK_DPD'] > 0).astype(int)
        pos_cash['pos_cash_paid_late_with_tolerance'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)
        last_k_trend_periods = [6, 12]
        last_k_agg_periods = [6, 12, 30]

        features = pd.DataFrame({'SK_ID_CURR': pos_cash['SK_ID_CURR'].unique()})
        groupby = pos_cash.groupby(['SK_ID_CURR'])
        func = partial(PosCashFeaturesOpenSolution.generate_features,
                       agg_periods=last_k_agg_periods,
                       trend_periods=last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=4).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        return features

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods):
        one_time = PosCashFeaturesOpenSolution.one_time_features(gr)
        all = PosCashFeaturesOpenSolution.all_installment_features(gr)
        agg = PosCashFeaturesOpenSolution.last_k_installment_features(gr, agg_periods)
        trend = PosCashFeaturesOpenSolution.trend_in_last_k_installment_features(gr, trend_periods)
        last = PosCashFeaturesOpenSolution.last_loan_features(gr)
        features = {**one_time, **all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def one_time_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
        features = {}

        features['pos_cash_remaining_installments'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1).astype(np.float64)
        features['pos_cash_completed_contracts'] = gr_['is_contract_status_completed'].agg('sum')

        return features

    @staticmethod
    def all_installment_features(gr):
        return PosCashFeaturesOpenSolution.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'SK_DPD', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['count', 'sum', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'std'],
                                         'last_loan_')

        return features


class PosCashFeaturesLeakyTargetEncoding(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(PosCash.get_df(conf), on="SK_ID_CURR", how="left")
        # fit with train data and transform with both date
        train_df = df[df['TARGET'].notnull()].copy()
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = TargetEncoder(cols=categorical_columns).fit(train_df, train_df['TARGET']).transform(df)
        df = df.groupby(by=['SK_ID_CURR'], as_index=False).agg({col: 'mean' for col in categorical_columns})
        return df[categorical_columns + ['SK_ID_CURR']].rename(
            columns={col: f"{col}_target_encode" for col in categorical_columns}
        )
