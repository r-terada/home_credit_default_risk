import gc
import pandas as pd
from tqdm import tqdm
from features import Feature


class GroupByAggregateFeature(Feature):
    _base_feature_class = None
    _groupby_aggregations = []

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        table = cls._base_feature_class.get_df(conf)
        features = pd.DataFrame({'SK_ID_CURR': table['SK_ID_CURR'].unique()})

        for groupby_cols, specs in tqdm(cls._groupby_aggregations):
            group_object = table.groupby(groupby_cols)
            for c in groupby_cols:
                if c not in features.columns:
                    features[c] = table[c]
            for select, agg in tqdm(specs):
                groupby_aggregate_name = cls._create_colname_from_specs(groupby_cols, select, agg)
                features = features.merge(group_object[select]
                                          .agg(agg)
                                          .reset_index()
                                          .rename(index=str,
                                                  columns={select: groupby_aggregate_name})
                                          [groupby_cols + [groupby_aggregate_name]],
                                          on=groupby_cols,
                                          how='left')
            features.drop(groupby_cols, axis=1)

        del table
        gc.collect()
        return features

    @classmethod
    def _create_colname_from_specs(cls, groupby_cols, select, agg):
        return '{}_{}_{}_{}'.format(cls._base_feature_class.__name__, '_'.join(groupby_cols), agg, select)
