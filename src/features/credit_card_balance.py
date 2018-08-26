import gc
import numpy as np
import pandas as pd

from category_encoders import TargetEncoder
from features import one_hot_encoder, Feature
from features.base import Base
from features.raw_data import CreditCardBalance
from features.feature_cleaner import clean_data


class CreditCardBalanceFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        cc = CreditCardBalance.get_df(conf)
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


class CreditCardBalanceFeaturesAntonova(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df_card = CreditCardBalance.get_df(conf)

        # Replace some outliers
        df_card.loc[df_card['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
        df_card.loc[df_card['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan

        # Some new features
        df_card['card missing'] = df_card.isnull().sum(axis = 1).values
        df_card['card SK_DPD - MONTHS_BALANCE'] = df_card['SK_DPD'] - df_card['MONTHS_BALANCE']
        df_card['card SK_DPD_DEF - MONTHS_BALANCE'] = df_card['SK_DPD_DEF'] - df_card['MONTHS_BALANCE']
        df_card['card SK_DPD - SK_DPD_DEF'] = df_card['SK_DPD'] - df_card['SK_DPD_DEF']

        df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card['AMT_RECIVABLE']
        df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card['AMT_RECEIVABLE_PRINCIPAL']
        df_card['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_RECIVABLE'] - df_card['AMT_RECEIVABLE_PRINCIPAL']

        df_card['card AMT_BALANCE - AMT_RECIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_RECIVABLE']
        df_card['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_BALANCE'] - df_card['AMT_RECEIVABLE_PRINCIPAL']
        df_card['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_TOTAL_RECEIVABLE']

        df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card['AMT_DRAWINGS_ATM_CURRENT']
        df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card['AMT_DRAWINGS_OTHER_CURRENT']
        df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card['AMT_DRAWINGS_POS_CURRENT']

        # Categorical features with One-Hot encode
        df_card, categorical = one_hot_encoder(df_card)

        # Aggregations for application set
        aggregations = {}
        for col in df_card.columns:
            aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
        df_card_agg = df_card.groupby('SK_ID_CURR').agg(aggregations)
        df_card_agg.columns = pd.Index(['CARD_' + e[0] + "_" + e[1].upper() for e in df_card_agg.columns.tolist()])

        # Count credit card lines
        df_card_agg['CARD_COUNT'] = df_card.groupby('SK_ID_CURR').size()
        del df_card
        gc.collect()

        return df_card_agg


class CreditCardBalanceFeaturesAntonovaCleaned(CreditCardBalanceFeaturesAntonova):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(CreditCardBalanceFeaturesAntonova.get_df(conf), on="SK_ID_CURR", how="left")
        return clean_data(df)


class CreditCardBalanceFeaturesLeakyTargetEncoding(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(CreditCardBalance.get_df(conf), on="SK_ID_CURR", how="left")
        # fit with train data and transform with both date
        train_df = df[df['TARGET'].notnull()].copy()
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = TargetEncoder(cols=categorical_columns).fit(train_df, train_df['TARGET']).transform(df)
        df = df.groupby(by=['SK_ID_CURR'], as_index=False).agg({col: 'mean' for col in categorical_columns})
        return df[categorical_columns + ['SK_ID_CURR']].rename(
            columns={col: f"{col}_target_encode" for col in categorical_columns}
        )
