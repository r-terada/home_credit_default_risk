import pandas as pd
from features.aggs import GroupByAggregateFeature
from features.raw_data import Application


COLS_TO_AGG = ['AMT_CREDIT',
               'AMT_ANNUITY',
               'AMT_INCOME_TOTAL',
               'AMT_GOODS_PRICE',
               'EXT_SOURCE_1',
               'EXT_SOURCE_2',
               'EXT_SOURCE_3',
               'OWN_CAR_AGE',
               'REGION_POPULATION_RELATIVE',
               'DAYS_REGISTRATION',
               'CNT_CHILDREN',
               'CNT_FAM_MEMBERS',
               'DAYS_ID_PUBLISH',
               'DAYS_BIRTH',
               'DAYS_EMPLOYED'
               ]
AGG_FUNCTIONS = ['min', 'mean', 'max', 'sum', 'var']
AGG_PAIRS = [(col, agg) for col in COLS_TO_AGG for agg in AGG_FUNCTIONS]

APPLICATION_AGGREGATION_RECIPIES = [
    (['NAME_EDUCATION_TYPE', 'CODE_GENDER'], AGG_PAIRS),
    (['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE'], AGG_PAIRS),
    (['NAME_FAMILY_STATUS', 'CODE_GENDER'], AGG_PAIRS),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),
]


class AggregateFeatureApplicationOpenSolution(GroupByAggregateFeature):
    _base_feature_class = Application
    _groupby_aggregations = APPLICATION_AGGREGATION_RECIPIES
