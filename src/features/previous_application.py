import gc
import numpy as np
import pandas as pd

from category_encoders import TargetEncoder
from features import one_hot_encoder, Feature, LargeFeature
from features.base import Base
from features.raw_data import PreviousApplication
from features.feature_cleaner import clean_data


class PreviousApplicationFeatures(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        prev = PreviousApplication.get_df(conf)
        prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
        # Add feature: value ask / value received percentage
        prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        # Previous applications numeric features
        num_aggregations = {
            'AMT_ANNUITY': ['max', 'mean'],
            'AMT_APPLICATION': ['max', 'mean'],
            'AMT_CREDIT': ['max', 'mean'],
            'APP_CREDIT_PERC': ['max', 'mean'],
            'AMT_DOWN_PAYMENT': ['max', 'mean'],
            'AMT_GOODS_PRICE': ['max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['max', 'mean'],
            'RATE_DOWN_PAYMENT': ['max', 'mean'],
            'DAYS_DECISION': ['max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
        }
        # Previous applications categorical features
        cat_aggregations = {}
        for cat in cat_cols:
            cat_aggregations[cat] = ['mean']

        prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
        prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
        # Previous Applications: Approved Applications - only numerical features
        approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
        approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
        # Previous Applications: Refused Applications - only numerical features
        refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
        refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
        del refused, refused_agg, approved, approved_agg, prev
        gc.collect()
        return prev_agg


class PreviousApplicationFeaturesOpenSolution(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        prev_applications = PreviousApplication.get_df(conf)
        features = pd.DataFrame({'SK_ID_CURR': prev_applications['SK_ID_CURR'].unique()})

        prev_app_sorted = prev_applications.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
        prev_app_sorted_groupby = prev_app_sorted.groupby(by=['SK_ID_CURR'])

        prev_app_sorted['previous_application_prev_was_approved'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
        g = prev_app_sorted_groupby['previous_application_prev_was_approved'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['previous_application_prev_was_refused'] = (
            prev_app_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        g = prev_app_sorted_groupby['previous_application_prev_was_refused'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'}, inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])['previous_application_prev_was_refused'].mean().reset_index()
        g.rename(index=str, columns={
            'previous_application_prev_was_refused': 'previous_application_fraction_of_refused_applications'},
            inplace=True)
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        prev_app_sorted['prev_applications_prev_was_revolving_loan'] = (
            prev_app_sorted['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
        g = prev_app_sorted.groupby(by=['SK_ID_CURR'])[
            'prev_applications_prev_was_revolving_loan'].last().reset_index()
        features = features.merge(g, on=['SK_ID_CURR'], how='left')

        for number in [1, 2, 3, 4, 5]:
            prev_applications_tail = prev_app_sorted_groupby.tail(number)

            tail_groupby = prev_applications_tail.groupby(by=['SK_ID_CURR'])

            g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(
                         number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
            g.rename(index=str,
                     columns={
                         'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(
                             number)},
                     inplace=True)
            features = features.merge(g, on=['SK_ID_CURR'], how='left')

        return features


class PreviousApplicationFeaturesAntonova(LargeFeature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df_prev = PreviousApplication.get_df(conf)

        # Replace some outliers
        df_prev.loc[df_prev['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
        df_prev.loc[df_prev['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
        df_prev[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                 'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace=True)

        # Some new features
        df_prev['prev missing'] = df_prev.isnull().sum(axis=1).values
        df_prev['prev AMT_APPLICATION / AMT_CREDIT'] = df_prev['AMT_APPLICATION'] / df_prev['AMT_CREDIT']
        df_prev['prev AMT_APPLICATION - AMT_CREDIT'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_CREDIT']
        df_prev['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_GOODS_PRICE']
        df_prev['prev AMT_GOODS_PRICE - AMT_CREDIT'] = df_prev['AMT_GOODS_PRICE'] - df_prev['AMT_CREDIT']
        df_prev['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = df_prev['DAYS_FIRST_DRAWING'] - df_prev['DAYS_FIRST_DUE']
        df_prev['prev DAYS_TERMINATION less -500'] = (df_prev['DAYS_TERMINATION'] < -500).astype(int)

        # Categorical features with One-Hot encode
        df_prev, categorical = one_hot_encoder(df_prev)

        # Aggregations for application set
        aggregations = {}
        for col in df_prev.columns:
            aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
        df_prev_agg = df_prev.groupby('SK_ID_CURR').agg(aggregations)
        df_prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in df_prev_agg.columns.tolist()])

        # Previous Applications: Approved Applications
        approved_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Approved'] == 1].groupby('SK_ID_CURR').agg(aggregations)
        approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
        df_prev_agg = df_prev_agg.join(approved_agg, how='left')
        del approved_agg
        gc.collect()

        # Previous Applications: Refused Applications
        refused_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Refused'] == 1].groupby('SK_ID_CURR').agg(aggregations)
        refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
        df_prev_agg = df_prev_agg.join(refused_agg, how='left')
        del refused_agg, df_prev
        gc.collect()

        return df_prev_agg


class PreviousApplicationFeaturesAntonovaCleaned(PreviousApplicationFeaturesAntonova):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(PreviousApplicationFeaturesAntonova.get_df(conf), on="SK_ID_CURR", how="left")
        return clean_data(df)


class PreviousApplicationFeaturesLeakyTargetEncoding(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(PreviousApplication.get_df(conf), on="SK_ID_CURR", how="left")
        # fit with train data and transform with both date
        train_df = df[df['TARGET'].notnull()].copy()
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = TargetEncoder(cols=categorical_columns).fit(train_df, train_df['TARGET']).transform(df)
        df = df.groupby(by=['SK_ID_CURR'], as_index=False).agg({col: 'mean' for col in categorical_columns})
        return df[categorical_columns + ['SK_ID_CURR']].rename(
            columns={col: f"{col}_target_encode" for col in categorical_columns}
        )
