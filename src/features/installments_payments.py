import os
import gc
import numpy as np
import pandas as pd

from functools import partial
from features import Feature, one_hot_encoder, parallel_apply, add_features_in_group, add_trend_feature, get_feature_names_by_period, safe_div
from features.base import Base
from features.raw_data import InstallmentsPayments
from features.feature_cleaner import clean_data


class InstallmentsPaymentsFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        ins = InstallmentsPayments.get_df(conf)
        ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
        # Percentage and difference paid in each installment (amount paid and installment value)
        ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
        ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
        # Days past due and days before due (no negative values)
        ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
        ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
        ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
        ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
        # Features: Perform aggregations
        aggregations = {
            'NUM_INSTALMENT_VERSION': ['nunique'],
            'DPD': ['max', 'mean', 'sum'],
            'DBD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['mean',  'var'],
            'PAYMENT_DIFF': ['mean', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        for cat in cat_cols:
            aggregations[cat] = ['mean']
        ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
        ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
        del ins
        gc.collect()
        return ins_agg


class InstallmentsPaymentsFeaturesOpenSolution(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        installments = InstallmentsPayments.get_df(conf)
        installments['installment_paid_late_in_days'] = installments['DAYS_ENTRY_PAYMENT'] - installments[
            'DAYS_INSTALMENT']
        installments['installment_paid_late'] = (installments['installment_paid_late_in_days'] > 0).astype(int)
        installments['installment_paid_over_amount'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
        installments['installment_paid_over'] = (installments['installment_paid_over_amount'] > 0).astype(int)

        features = pd.DataFrame({'SK_ID_CURR': installments['SK_ID_CURR'].unique()})
        groupby = installments.groupby(['SK_ID_CURR'])

        last_k_trend_periods = [10, 50, 100, 500]
        last_k_agg_periods = [1, 5, 10, 20, 50, 100]
        last_k_agg_period_fractions = [(5, 20), (5, 50), (10, 50), (10, 100), (20, 100)]
        func = partial(InstallmentsPaymentsFeaturesOpenSolution.generate_features,
                       agg_periods=last_k_agg_periods,
                       period_fractions=last_k_agg_period_fractions,
                       trend_periods=last_k_trend_periods)
        g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=4).reset_index()
        features = features.merge(g, on='SK_ID_CURR', how='left')

        # dtype of "pos_cash_remaining_installments" is object, don't know why...
        features["pos_cash_remaining_installments"] = features["pos_cash_remaining_installments"].astype(np.float64)
        return features

    @staticmethod
    def generate_features(gr, agg_periods, trend_periods, period_fractions):
        all = InstallmentsPaymentsFeaturesOpenSolution.all_installment_features(gr)
        agg = InstallmentsPaymentsFeaturesOpenSolution.last_k_installment_features_with_fractions(gr,
                                                                                                  agg_periods,
                                                                                                  period_fractions)
        trend = InstallmentsPaymentsFeaturesOpenSolution.trend_in_last_k_installment_features(gr, trend_periods)
        last = InstallmentsPaymentsFeaturesOpenSolution.last_loan_features(gr)
        features = {**all, **agg, **trend, **last}
        return pd.Series(features)

    @staticmethod
    def all_installment_features(gr):
        return InstallmentsPaymentsFeaturesOpenSolution.last_k_installment_features(gr, periods=[10e16])

    @staticmethod
    def last_k_installment_features_with_fractions(gr, periods, period_fractions):
        features = InstallmentsPaymentsFeaturesOpenSolution.last_k_installment_features(gr, periods)

        for short_period, long_period in period_fractions:
            short_feature_names = get_feature_names_by_period(features, short_period)
            long_feature_names = get_feature_names_by_period(features, long_period)

            for short_feature, long_feature in zip(short_feature_names, long_feature_names):
                old_name_chunk = '_{}_'.format(short_period)
                new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
                fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
                features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
        return features

    @staticmethod
    def last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            if period > 10e10:
                period_name = 'all_installment_'
                gr_period = gr_.copy()
            else:
                period_name = 'last_{}_'.format(period)
                gr_period = gr_.iloc[:period]

            features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'iqr'],
                                             period_name)

            features = add_features_in_group(features, gr_period, 'installment_paid_late_in_days',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_late',
                                             ['count', 'mean'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over_amount',
                                             ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'iqr'],
                                             period_name)
            features = add_features_in_group(features, gr_period, 'installment_paid_over',
                                             ['count', 'mean'],
                                             period_name)
        return features

    @staticmethod
    def trend_in_last_k_installment_features(gr, periods):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

        features = {}
        for period in periods:
            gr_period = gr_.iloc[:period]

            features = add_trend_feature(features, gr_period,
                                         'installment_paid_late_in_days', '{}_period_trend_'.format(period)
                                         )
            features = add_trend_feature(features, gr_period,
                                         'installment_paid_over_amount', '{}_period_trend_'.format(period)
                                         )
        return features

    @staticmethod
    def last_loan_features(gr):
        gr_ = gr.copy()
        gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
        last_installment_id = gr_['SK_ID_PREV'].iloc[0]
        gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

        features = {}
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_late',
                                         ['count', 'mean'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
        features = add_features_in_group(features, gr_,
                                         'installment_paid_over',
                                         ['count', 'mean'],
                                         'last_loan_')
        return features


class InstallmentsPaymentsFeaturesAntonova(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df_ins = InstallmentsPayments.get_df(conf)

        # Replace some outliers
        df_ins.loc[df_ins['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
        df_ins.loc[df_ins['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan

        # Some new features
        df_ins['ins DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT'] = df_ins['DAYS_ENTRY_PAYMENT'] - df_ins['DAYS_INSTALMENT']
        df_ins['ins NUM_INSTALMENT_NUMBER_100'] = (df_ins['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
        df_ins['ins DAYS_INSTALMENT more NUM_INSTALMENT_NUMBER'] = (df_ins['DAYS_INSTALMENT'] > df_ins['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)
        df_ins['ins AMT_INSTALMENT - AMT_PAYMENT'] = df_ins['AMT_INSTALMENT'] - df_ins['AMT_PAYMENT']
        df_ins['ins AMT_PAYMENT / AMT_INSTALMENT'] = df_ins['AMT_PAYMENT'] / df_ins['AMT_INSTALMENT']

        # Categorical features with One-Hot encode
        df_ins, categorical = one_hot_encoder(df_ins)

        # Aggregations for application set
        aggregations = {}
        for col in df_ins.columns:
            aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
        df_ins_agg = df_ins.groupby('SK_ID_CURR').agg(aggregations)
        df_ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in df_ins_agg.columns.tolist()])

        # Count installments lines
        df_ins_agg['INSTAL_COUNT'] = df_ins.groupby('SK_ID_CURR').size()
        del df_ins
        gc.collect()

        return df_ins_agg


class InstallmentsPaymentsFeaturesAntonovaCleaned(InstallmentsPaymentsFeaturesAntonova):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(InstallmentsPaymentsFeaturesAntonova.get_df(conf), on="SK_ID_CURR", how="left")
        return clean_data(df)
