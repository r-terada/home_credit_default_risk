import os
import gc
import numpy as np
import pandas as pd

from typing import Tuple
from features import one_hot_encoder, Feature
from features.raw_data import Application


class ApplicationFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Application.get_df(conf)
        df = ApplicationFeatures._clean_data(df)
        df = ApplicationFeatures._features_from_kernel(df)
        df = ApplicationFeatures._binarize_features(df)
        df, _cat_cols = one_hot_encoder(df, True)
        return ApplicationFeatures._filter_features(df)

    @staticmethod
    def _clean_data(df):
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
        return df

    @staticmethod
    def _binarize_features(df):
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            df[bin_feature], _uniques = pd.factorize(df[bin_feature])

        return df

    @staticmethod
    def _features_from_kernel(df):
        docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
        live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

        inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

        df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
        df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
        df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
        df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
        df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
        df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
        df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
        df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
        df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
        df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
        df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        return df

    @staticmethod
    def _filter_features(df):
        dropcolum = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4',
                     'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                     'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                     'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
                     'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                     'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19',
                     'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'TARGET']
        return df.drop(dropcolum, axis=1)


class ApplicationFeaturesOpenSolution(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        X = Application.get_df(conf)
        X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
        X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
        X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
        X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
        X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
        X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
        X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
        X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
        X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
        X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
        X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
        X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
        X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
        X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
        X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
        X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
        X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        X['short_employment'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
        X['young_age'] = (X['DAYS_BIRTH'] < -14000).astype(int)

        return X
