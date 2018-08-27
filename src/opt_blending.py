import os
import gc
import sys
import json
import click
import pickle
import random
import sklearn
import hyperopt
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from datetime import datetime
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK

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

    if "drop_features_list_file" in conf.options:
        with open(conf.options.drop_features_list_file, "r") as fp:
            line = fp.read()
            feature_to_drop = eval(line)
        print(f"drop columns in {conf.options.drop_features_list_file}")
        df = df.drop(feature_to_drop, axis=1)

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
@click.option('--num_opt_eval', type=int, default=200)
def main(config_file, num_opt_eval):
    conf = read_config(config_file)
    print("config:")
    pprint(conf)

    train_df, test_df = get_train_test(conf)
    feats = [f for f in train_df.columns if f not in ([
        'TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'
    ])]
    print(f"use {len(feats)} features.")

    pbar = tqdm(total=num_opt_eval, desc="optimize blend")

    def objective(params):
        pbar.update()
        warnings.simplefilter('ignore')

        s = sum([v for k, v in params.items() if k in feats])
        for p in feats:
            params[p] = params[p] / s

        test_pred_proba = pd.Series(np.zeros(train_df.shape[0]), index=train_df.index)

        for f in feats:
            test_pred_proba += (train_df[f] ** params["power"]) * params[f]

        score = roc_auc_score(train_df['TARGET'], test_pred_proba)
        pbar.write(f"power {params['power']}: {score:.6f}")
        return {'loss': -1.0 * score, 'status': STATUS_OK}

    parameter_space = {"power": hp.quniform("power", 1, 8, 1)}
    for c in feats:
        parameter_space[c] = hp.quniform(c, 0, 1, 0.001)

    print("====== optimize blending parameters ======")
    trials = hyperopt.Trials()
    best = fmin(objective, parameter_space, algo=tpe.suggest,
                max_evals=num_opt_eval, trials=trials, verbose=1)
    pbar.close()

    s = sum([v for k, v in best.items() if k in feats])
    for p in feats:
        best[p] = best[p] / s

    print("====== best weights to blend ======")
    pprint(best)
    # for key, val in best.items():
    #     print(f"    {key}: {val}")
    print("============= best score =============")
    best_score = -1.0 * trials.best_trial['result']['loss']
    print(best_score)

    output_directory = os.path.join(
        conf.dataset.output_directory,
        f"{datetime.now().strftime('%m%d%H%M')}_{conf.config_file_name}_{best_score:.6f}"
    )
    print(f"write results to {output_directory}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(os.path.join(output_directory, "result.json"), "w") as fp:
        results = {"weights": best, "score": best_score, "eval_num": num_opt_eval}
        print(results)
        json.dump(results, fp, indent=2)

    pickle.dump(trials.trials, open(
        os.path.join(
            output_directory,
            f'trials.pkl'
        ), 'wb')
    )

    sub_preds = np.zeros(test_df.shape[0])
    for f in feats:
        sub_preds += (test_df[f] ** best["power"]) * best[f]
    test_df['TARGET'] = sub_preds
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

if __name__ == '__main__':
    main()
