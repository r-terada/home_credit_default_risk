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

    if "stacking_features" in conf:
        StackingFeaturesWithPasses.set_result_dirs(conf.stacking_features)
        f = StackingFeaturesWithPasses.get_df(conf)
        df = df.merge(f, how='left', on='SK_ID_CURR')

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
        for f in feats:
            params[f] = params[f] / s

        ranks = pd.DataFrame(columns=feats)
        for f in feats:
            ranks[f] = train_df[f].rank(method='min') * params[f]
        ranks['Average'] = ranks.mean(axis=1)
        ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
        score = roc_auc_score(train_df['TARGET'], ranks['Scaled Rank'])
        pbar.write(f"{score:.6f}")
        return {'loss': -1.0 * score, 'status': STATUS_OK}

    parameter_space = {}
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

    ranks = pd.DataFrame(columns=feats)
    for f in feats:
        ranks[f] = test_df[f].rank(method='min') * best[f]
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    test_df['TARGET'] = ranks['Scaled Rank']
    test_df[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

if __name__ == '__main__':
    main()
