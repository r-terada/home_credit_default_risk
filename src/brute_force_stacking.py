import os
import gc
import sys
import glob
import click
import random
import sklearn
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from datetime import datetime

from config import read_config, KEY_FEATURE_MAP, KEY_MODEL_MAP
from utils import timer, reduce_mem_usage
from models import LightGBM
from features.base import Base
from features.feature_cleaner import clean_data
from features.stacking import StackingFeaturesWithPasses

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_candidates(conf):
    dir_names = []
    for path in glob.glob(f"{conf.dataset.output_directory}/**", recursive=True):
        if not os.path.isdir(path):
            continue

        if os.path.exists(os.path.join(path, "oof_predictions.csv")) and \
           os.path.exists(os.path.join(path, "submission.csv")) and \
           float(path.split("_")[-1]) > conf.threshold and \
           "stacking" not in path:
            dir_names.append(path)

    return list(set(dir_names))


def get_train_test(conf):
    return split_train_test(get_df(conf, conf.stacking_features))


def get_df(conf, stacking_features):
    df = Base.get_df(conf)  # pd.DataFrame
    if stacking_features:
        StackingFeaturesWithPasses.set_result_dirs(stacking_features)
        f = StackingFeaturesWithPasses.get_df(conf)
        df = df.merge(f, how='left', on='SK_ID_CURR')
    return df


def split_train_test(df):
    train_df = df[df['TARGET'].notnull()].copy()
    test_df = df[df['TARGET'].isnull()].copy()
    del df
    gc.collect()
    return train_df, test_df


@click.command()
@click.option('--config_file', type=str, default='./configs/stacking_brute_force.json')
def main(config_file):
    conf = read_config(config_file)
    candidates = get_candidates(conf)

    if conf.stacking_features:
        best_features = list(conf.stacking_features[:])
        for f in best_features:
            for c in candidates[:]:
                if f in c:
                    candidates.remove(c)
    else:
        best_features = []

    print("candidates")
    print('\n'.join(candidates))
    print(f"search best stackers from {len(candidates)} outputs")

    print(f"conf:")
    pprint(conf)

    best_score_whole = 0.0
    while True:
        best_feature_loop = ""
        best_score_loop = 0.0

        print(f"add feature to {best_features}")
        base_df = get_df(conf, best_features)
        for feature in tqdm(candidates):
            print(f"current best {best_score_loop} : {best_feature_loop}")
            cur_df = base_df.merge(get_df(conf, [feature]).drop(["TARGET", "index"], axis=1), on='SK_ID_CURR', how='left')
            train_df, test_df = split_train_test(cur_df)
            feats = [f for f in train_df.columns if f not in ([
                'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
            ])]
            print(f"add {feature}")
            model = KEY_MODEL_MAP[conf.model.name]()
            score = model.train_and_predict_kfold(
                train_df,
                test_df,
                feats,
                'TARGET',
                conf,
                save_result=False
            )
            if score > best_score_loop:
                best_score_loop = score
                best_feature_loop = feature

        if best_score_loop > best_score_whole:
            best_score_whole = best_score_loop
            best_features.append(best_feature_loop)
            candidates.remove(best_feature_loop)
            print(f"=== current best score: {best_score_whole} ===")
            print(f"features: {best_features}")
        else:
            print(f"no improvement. break.")
            break

    print(f"best score {best_score_whole}")
    print(f"{best_features}")

    conf.stacking_features = best_features
    train_df, test_df = get_train_test(conf)
    feats = [f for f in train_df.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    model = KEY_MODEL_MAP[conf.model.name]()
    score = model.train_and_predict_kfold(
        train_df,
        test_df,
        feats,
        'TARGET',
        conf,
        save_result=True
    )

if __name__ == '__main__':
    main()
