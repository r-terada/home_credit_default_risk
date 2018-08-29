import os
import gc
import sys
import click
import random
import sklearn
import numpy as np
import pandas as pd
from pprint import pprint

from config import read_config, KEY_FEATURE_MAP, KEY_MODEL_MAP
from utils import timer, reduce_mem_usage
from models import LightGBM
from features.base import Base
from features.feature_cleaner import clean_data
from features.stacking import StackingFeaturesWithPasses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_train_test(conf):
    df = Base.get_df(conf)  # pd.DataFrame

    feature_classes = [KEY_FEATURE_MAP[key] for key in conf.features]
    for feature in feature_classes:
        with timer(f"process {feature.__name__}"):
            f = feature.get_df(conf)
            if "drop_duplicate_column_on_merge" in conf.options and conf.options.drop_duplicate_column_on_merge:
                cols_to_drop = [c for c in f.columns if (c in df.columns) and (c != 'SK_ID_CURR')]
                if cols_to_drop:
                    print(f"drop columns: {cols_to_drop}")
                    f = f.drop(cols_to_drop, axis=1)
            if "reduce_mem_usage" in conf.options and conf.options.reduce_mem_usage:
                with timer("reduce_mem_usaga"):
                    f = reduce_mem_usage(f)
            df = df.merge(f, how='left', on='SK_ID_CURR')
            del f
            gc.collect()

    if "stacking_features" in conf:
        StackingFeaturesWithPasses.set_result_dirs(conf.stacking_features)
        f = StackingFeaturesWithPasses.get_df(conf)
        df = df.merge(f, how='left', on='SK_ID_CURR')

    if "clean_data" in conf.options and conf.options.clean_data:
        with timer("clean_data"):
            df = clean_data(df)

    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    del df
    gc.collect()
    return train_df, test_df


@click.command()
@click.option('--config_file', type=str, default='./configs/lgbm_0.json')
@click.option('--full_train', is_flag=True, default=False)
def main(config_file, full_train):
    conf = read_config(config_file)
    print("config:")
    pprint(conf)

    train_df_full, test_df_full = get_train_test(conf)

    for f in [
        "misc/gain_corr_under_threshold_99.txt",
        "misc/gain_corr_under_threshold_95.txt",
        "misc/split_corr_under_threshold_99.txt",
        "misc/split_corr_under_threshold_95.txt",
        "misc/gain_corr_under_threshold_10.txt",
        "misc/gain_corr_under_threshold_20.txt",
        "misc/gain_corr_under_threshold_30.txt",
        "misc/gain_corr_under_threshold_40.txt",
        "misc/gain_corr_under_threshold_50.txt",
        "misc/gain_corr_under_threshold_60.txt",
        "misc/gain_corr_under_threshold_70.txt",
        "misc/gain_corr_under_threshold_80.txt",
        "misc/split_corr_under_threshold_10.txt",
        "misc/split_corr_under_threshold_20.txt",
        "misc/split_corr_under_threshold_30.txt",
        "misc/split_corr_under_threshold_40.txt",
        "misc/split_corr_under_threshold_50.txt",
        "misc/split_corr_under_threshold_60.txt",
        "misc/split_corr_under_threshold_70.txt",
        "misc/split_corr_under_threshold_80.txt",
        "misc/split_corr_under_threshold_90.txt",
    ]:
        config.config_file_name = f"{config.config_file_name}_{os.path.basename(f).split(".")[0]}"
        with open(f, "r") as fp:
            line = fp.read()
            feature_to_drop = eval(line)
        print(f"drop columns in {f}")
        train_df = train_df_full.drop(feature_to_drop, axis=1)
        test_df = test_df_full.drop(feature_to_drop, axis=1)

        feats = [f for f in train_df.columns if f not in ([
            'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
        ])]
        print(f"use {len(feats)} features.")
        print(train_df.shape)

        model = KEY_MODEL_MAP[conf.model.name]()
        with timer(f"train with {model.__class__.__name__}"):
            if not full_train:
                model.train_and_predict_kfold(
                    train_df,
                    test_df,
                    feats,
                    'TARGET',
                    conf
                )

            else:
                model.train_and_predict(
                    train_df,
                    test_df,
                    feats,
                    'TARGET',
                    conf
                )


if __name__ == '__main__':
    main()
