import os
import gc
import sys
import glob
import click
import random
import sklearn
import numpy as np
import pandas as pd
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

    return dir_names


def get_train_test(conf):
    df = Base.get_df(conf)  # pd.DataFrame
    StackingFeaturesWithPasses.set_result_dirs(conf.stacking_features)
    f = StackingFeaturesWithPasses.get_df(conf)
    df = df.merge(f, how='left', on='SK_ID_CURR')
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
    print("candidates")
    print('\n'.join(candidates))

    print(f"conf:")
    pprint(conf)

    best_features = []
    best_score_whole = 0.0
    next_candidates = None
    while True:
        best_feature_loop = ""
        best_score_loop = 0.0

        if next_candidates:
            candidates = next_candidates[:]

        next_candidates = []
        print(f"search best stacker from {len(candidates)} outputs")
        for feature in candidates:
            conf.stacking_features = best_features + [feature]
            train_df, test_df = get_train_test(conf)
            feats = [f for f in train_df.columns if f not in ([
                'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
            ])]
            print([os.path.basename(f) for f in conf.stacking_features])
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

            if score >= best_score_whole:
                next_candidates.append(feature)

        if best_score_loop > best_score_whole:
            best_score_whole = best_score_loop
            best_features.append(best_feature_loop)
            next_candidates.remove(best_feature_loop)
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
