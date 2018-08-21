import os
import gc
import numpy as np
import pandas as pd

from typing import Tuple
from category_encoders import TargetEncoder
from features import one_hot_encoder, Feature
from features.raw_data import Application


class ApplicationFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Application.get_df(conf)
        df = ApplicationFeatures._features_from_kernel(df)
        df = ApplicationFeatures._binarize_features(df)
        df, _cat_cols = one_hot_encoder(df, True)
        return ApplicationFeatures._filter_features(df)

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
        app = Application.get_df(conf)
        features = pd.DataFrame({'SK_ID_CURR': app['SK_ID_CURR'].unique()})
        features['annuity_income_percentage'] = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL']
        features['car_to_birth_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_BIRTH']
        features['car_to_employ_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_EMPLOYED']
        features['children_ratio'] = app['CNT_CHILDREN'] / app['CNT_FAM_MEMBERS']
        features['credit_to_annuity_ratio'] = app['AMT_CREDIT'] / app['AMT_ANNUITY']
        features['credit_to_goods_ratio'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']
        features['credit_to_income_ratio'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']
        features['days_employed_percentage'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']
        features['income_credit_percentage'] = app['AMT_INCOME_TOTAL'] / app['AMT_CREDIT']
        features['income_per_child'] = app['AMT_INCOME_TOTAL'] / (1 + app['CNT_CHILDREN'])
        features['income_per_person'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']
        features['payment_rate'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']
        features['phone_to_birth_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_BIRTH']
        features['phone_to_employ_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_EMPLOYED']
        features['external_sources_weighted'] = app.EXT_SOURCE_1 * 2 + app.EXT_SOURCE_2 * 3 + app.EXT_SOURCE_3 * 4
        features['cnt_non_child'] = app['CNT_FAM_MEMBERS'] - app['CNT_CHILDREN']
        features['child_to_non_child_ratio'] = app['CNT_CHILDREN'] / features['cnt_non_child']
        features['income_per_non_child'] = app['AMT_INCOME_TOTAL'] / features['cnt_non_child']
        features['credit_per_person'] = app['AMT_CREDIT'] / app['CNT_FAM_MEMBERS']
        features['credit_per_child'] = app['AMT_CREDIT'] / (1 + app['CNT_CHILDREN'])
        features['credit_per_non_child'] = app['AMT_CREDIT'] / features['cnt_non_child']
        for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
            features['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                app[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

        features['short_employment'] = (app['DAYS_EMPLOYED'] < -2000).astype(int)
        features['young_age'] = (app['DAYS_BIRTH'] < -14000).astype(int)

        return features


class ApplicationFeaturesLeakyTargetEncoding(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Application.get_df(conf)
        # fit with train data and transform with both date
        train_df = df[df['TARGET'].notnull()].copy()
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = TargetEncoder(cols=categorical_columns).fit(df, df['TARGET']).transform(df)
        return df[categorical_columns + ['SK_ID_CURR']].rename(
            columns={col: f"{col}_target_encode" for col in categorical_columns}
        )
