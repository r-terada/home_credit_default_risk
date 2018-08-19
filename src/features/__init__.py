import os
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from scipy.stats import kurtosis, iqr, skew
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0


def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


class Feature:

    @classmethod
    def get_df(cls, conf) -> pd.DataFrame:
        pkl_fpath = cls._file_path(conf)
        if os.path.exists(pkl_fpath):
            return cls._read_pickle(pkl_fpath)
        else:
            print("no pickled file. create feature")
            df = cls._create_feature(conf)
            print(f"save to {pkl_fpath}")
            cls._save_as_pickled_object(df, pkl_fpath)
            return df

    @classmethod
    def _file_path(cls, conf) -> str:
        return os.path.join(
            conf.dataset.cache_directory,
            "features",
            f"{cls.__name__}.pkl"
        )

    @classmethod
    def _read_pickle(cls, pkl_fpath) -> pd.DataFrame:
        return pd.read_pickle(pkl_fpath)

    @classmethod
    def _save_as_pickled_object(cls, df, pkl_fpath) -> None:
        df.to_pickle(pkl_fpath)

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        raise NotImplementedError


