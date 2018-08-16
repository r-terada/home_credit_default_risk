import os
import gc
import pandas as pd


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


class Feature:

    @classmethod
    def get_df(cls, conf) -> pd.DataFrame:
        pkl_fpath = os.path.join(
            conf.dataset.cache_directory,
            "features",
            f"{cls.__name__}.pkl"
        )
        if os.path.exists(pkl_fpath):
            return cls._read_pickle(pkl_fpath)
        else:
            print("no pickled file. create feature")
            df = cls._create_feature(conf)
            print(f"save to {pkl_fpath}")
            df.to_pickle(pkl_fpath)
            return df

    @classmethod
    def _read_pickle(cls, pkl_fpath) -> pd.DataFrame:
        return pd.read_pickle(pkl_fpath)

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        raise NotImplementedError
