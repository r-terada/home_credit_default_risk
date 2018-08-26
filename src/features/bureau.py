import os
import gc
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder

from features import one_hot_encoder, Feature, LargeFeature
from features.base import Base
from features.feature_cleaner import clean_data
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


class BureauFeaturesAntonova(LargeFeature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df_bureau_b = BureauBalance.get_df(conf)

        # Some new features in bureau_balance set
        tmp = df_bureau_b[['SK_ID_BUREAU', 'STATUS']].groupby('SK_ID_BUREAU')
        tmp_last = tmp.last()
        tmp_last.columns = ['First_status']
        df_bureau_b = df_bureau_b.join(tmp_last, how='left', on='SK_ID_BUREAU')
        tmp_first = tmp.first()
        tmp_first.columns = ['Last_status']
        df_bureau_b = df_bureau_b.join(tmp_first, how='left', on='SK_ID_BUREAU')
        del tmp, tmp_first, tmp_last
        gc.collect()

        tmp = df_bureau_b[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()
        tmp = tmp.apply(abs)
        tmp.columns = ['Month']
        df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
        del tmp
        gc.collect()

        tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == 'C', ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
            .groupby('SK_ID_BUREAU').last()
        tmp = tmp.apply(abs)
        tmp.columns = ['When_closed']
        df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
        del tmp
        gc.collect()

        df_bureau_b['Month_closed_to_end'] = df_bureau_b['Month'] - df_bureau_b['When_closed']

        for c in range(6):
            tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == str(c), ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
                             .groupby('SK_ID_BUREAU').count()
            tmp.columns = ['DPD_' + str(c) + '_cnt']
            df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
            df_bureau_b['DPD_' + str(c) + ' / Month'] = df_bureau_b['DPD_' + str(c) + '_cnt'] / df_bureau_b['Month']
            del tmp
            gc.collect()
        df_bureau_b['Non_zero_DPD_cnt'] = df_bureau_b[['DPD_1_cnt', 'DPD_2_cnt', 'DPD_3_cnt', 'DPD_4_cnt', 'DPD_5_cnt']].sum(axis=1)

        df_bureau_b, bureau_b_cat = one_hot_encoder(df_bureau_b)

        # Bureau balance: Perform aggregations
        aggregations = {}
        for col in df_bureau_b.columns:
            aggregations[col] = ['mean'] if col in bureau_b_cat else ['min', 'max', 'size']
        df_bureau_b_agg = df_bureau_b.groupby('SK_ID_BUREAU').agg(aggregations)
        df_bureau_b_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_bureau_b_agg.columns.tolist()])
        del df_bureau_b
        gc.collect()

        df_bureau = Bureau.get_df(conf)

        # Replace\remove some outliers in bureau set
        df_bureau.loc[df_bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
        df_bureau.loc[df_bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
        df_bureau.loc[df_bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
        df_bureau.loc[df_bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
        df_bureau.loc[df_bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
        df_bureau.loc[(df_bureau['DAYS_CREDIT_UPDATE'] > 0) | (df_bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
        df_bureau.loc[df_bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan

        df_bureau.drop(df_bureau[df_bureau['DAYS_ENDDATE_FACT'] < df_bureau['DAYS_CREDIT']].index, inplace=True)

        # Some new features in bureau set
        df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau['AMT_CREDIT_SUM_DEBT']
        df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau['AMT_CREDIT_SUM_LIMIT']
        df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau['AMT_CREDIT_SUM_OVERDUE']

        df_bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = df_bureau['DAYS_CREDIT'] - df_bureau['CREDIT_DAY_OVERDUE']
        df_bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_CREDIT_ENDDATE']
        df_bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_ENDDATE_FACT']
        df_bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT_ENDDATE'] - df_bureau['DAYS_ENDDATE_FACT']
        df_bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT_UPDATE'] - df_bureau['DAYS_CREDIT_ENDDATE']

        # Categorical features with One-Hot encode
        df_bureau, bureau_cat = one_hot_encoder(df_bureau)

        # Bureau balance: merge with bureau.csv
        df_bureau = df_bureau.join(df_bureau_b_agg, how='left', on='SK_ID_BUREAU')
        df_bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
        del df_bureau_b_agg
        gc.collect()

        # Bureau and bureau_balance aggregations for application set
        categorical = bureau_cat + bureau_b_cat
        aggregations = {}
        for col in df_bureau.columns:
            aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
        df_bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(aggregations)
        df_bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in df_bureau_agg.columns.tolist()])

        # Bureau: Active credits
        active_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Active'] == 1].groupby('SK_ID_CURR').agg(aggregations)
        active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
        df_bureau_agg = df_bureau_agg.join(active_agg, how='left')
        del active_agg
        gc.collect()

        # Bureau: Closed credits
        closed_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Closed'] == 1].groupby('SK_ID_CURR').agg(aggregations)
        closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
        df_bureau_agg = df_bureau_agg.join(closed_agg, how='left')
        del closed_agg, df_bureau
        gc.collect()

        return df_bureau_agg


class BureauFeaturesAntonovaCleaned(BureauFeaturesAntonova):
    @classmethod
    def _create_feature(cls, conf) ->pd.DataFrame:
        base = Base.get_df(conf)
        df = BureauFeaturesAntonova.get_df(conf)
        # clean data needs target info
        return clean_data(base.merge(df, on='SK_ID_CURR', how='left'))


class BureauFeaturesLeakyTargetEncoding(Feature):
    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Base.get_df(conf)
        df = df.merge(Bureau.get_df(conf), on="SK_ID_CURR", how="left")
        # fit with train data and transform with both date
        train_df = df[df['TARGET'].notnull()].copy()
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
        df = TargetEncoder(cols=categorical_columns).fit(train_df, train_df['TARGET']).transform(df)
        df = df.groupby(by=['SK_ID_CURR'], as_index=False).agg({col: 'mean' for col in categorical_columns})
        return df[categorical_columns + ['SK_ID_CURR']].rename(
            columns={col: f"{col}_target_encode" for col in categorical_columns}
        )
