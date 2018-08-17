import os
import gc
import sys
import click
import random
import sklearn
import numpy as np
import pandas as pd
from pprint import pprint

from config import read_config, KEY_FEATURE_MAP
from utils import timer
from models import LightGBM
from features.base import Base

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_train_test(conf):
    df = Base.get_df(conf)  # pd.DataFrame

    feature_classes = [KEY_FEATURE_MAP[key] for key in conf.features]
    for feature in feature_classes:
        with timer(f"process {feature.__name__}"):
            f = feature.get_df(conf)
            df = df.merge(f, how='left', on='SK_ID_CURR')
            del f
            gc.collect()

    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    del df
    gc.collect()
    return train_df, test_df


@click.command()
@click.option('--config_file', type=str, default='./configs/lgbm_0.json')
@click.option('--debug', is_flag=True, default=False)
def main(config_file, debug):
    conf = read_config(config_file)
    print("config:")
    pprint(conf)
    train_df, test_df = get_train_test(conf)
    feats = [f for f in train_df.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    print(f"use {len(feats)} features.")
    model = LightGBM()
    with timer("train with lgbm"):
        model.train_and_predict_kfold(
            train_df,
            test_df,
            feats,
            'TARGET',
            conf
        )


if __name__ == '__main__':
    main()