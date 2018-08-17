import os
import gc
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from attrdict import AttrDict
from typing import List
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class Model:

    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> None:
        raise NotImplementedError


class LightGBM(Model):

    # override
    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> None:
        # prepare
        x_train = train[feats]
        y_train = train[target]
        x_test = test[feats]
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])
        feature_importance_df = pd.DataFrame()
        trials = {}

        # cv
        folds = StratifiedKFold(**conf.model.kfold_params)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
            start_time = time.time()
            train_x, train_y = x_train.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
            valid_x, valid_y = x_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx].copy()

            clf = LGBMClassifier(**conf.model.clf_params)
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], **conf.model.train_params)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(x_test, num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

            score = roc_auc_score(valid_y, oof_preds[valid_idx])
            print(f'Fold {(n_fold + 1):2d} AUC : {score:.6f}')

            fold_importance_df = pd.DataFrame.from_dict({
                "feature": feats,
                "importance": clf.feature_importances_,
                "fold": n_fold + 1
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            trials[f"Fold{n_fold + 1}"] = {
                "val_score": score,
                "best_iteration": clf._best_iteration,
                "elapsed_time": time.time() - start_time,
                "feature_importance_top10": {
                    row["feature"]: row["importance"] for i, row in fold_importance_df.sort_values("importance", ascending=False).head(10).iterrows()
                }
            }

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        score = roc_auc_score(y_train, oof_preds)
        print(f'Full AUC score {score:.6f}')

        trials["Full"] = {
            "score": score,
            "feature_importance_top10": {
                feature: row["importance"] for feature, row in
                feature_importance_df.groupby("feature").agg({"importance": "mean"}).sort_values("importance", ascending=False).head(10).iterrows()
            }
        }

        # write results
        output_directory = os.path.join(
            conf.dataset.output_directory,
            f"{datetime.now().strftime('%m%d%H%M')}_{conf.config_file_name}_{score:.6f}"
        )
        print(f"write results to {output_directory}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(os.path.join(output_directory, "result.json"), "w") as fp:
            results = {"trials": trials, "config": conf, "features_detail": {"number": len(feats), "names": feats}}
            json.dump(results, fp, indent=2)

        feature_importance_fname = os.path.join(output_directory, f'feature_importance.pkl')
        feature_importance_df.to_pickle(feature_importance_fname)

        del feature_importance_df
        gc.collect()

        test['TARGET'] = sub_preds
        test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)


class XGBoost(Model):

    # override
    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> None:
        # prepare
        x_train = train[feats]
        y_train = train[target]
        x_test = test[feats]
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])
        feature_importance_df = pd.DataFrame()
        trials = {}

        # cv
        folds = StratifiedKFold(**conf.model.kfold_params)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
            start_time = time.time()
            train_x, train_y = x_train.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
            valid_x, valid_y = x_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx].copy()

            clf = XGBClassifier(**conf.model.clf_params)
            clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], **conf.model.train_params)

            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            sub_preds += clf.predict_proba(x_test)[:, 1] / folds.n_splits

            score = roc_auc_score(valid_y, oof_preds[valid_idx])
            print(f'Fold {(n_fold + 1):2d} AUC : {score:.6f}')

            fold_importance_df = pd.DataFrame.from_dict({
                "feature": feats,
                "importance": clf.feature_importances_,
                "fold": n_fold + 1
            })
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            trials[f"Fold{n_fold + 1}"] = {
                "val_score": score,
                "best_iteration": clf.best_iteration,
                "elapsed_time": time.time() - start_time,
                "feature_importance_top10": {
                    row["feature"]: float(row["importance"]) for i, row in fold_importance_df.sort_values("importance", ascending=False).head(10).iterrows()
                }
            }

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        score = roc_auc_score(y_train, oof_preds)
        print(f'Full AUC score {score:.6f}')

        trials["Full"] = {
            "score": score,
            "feature_importance_top10": {
                feature: float(row["importance"]) for feature, row in
                feature_importance_df.groupby("feature").agg({"importance": "mean"}).sort_values("importance", ascending=False).head(10).iterrows()
            }
        }

        # write results
        output_directory = os.path.join(
            conf.dataset.output_directory,
            f"{datetime.now().strftime('%m%d%H%M')}_{conf.config_file_name}_{score:.6f}"
        )
        print(f"write results to {output_directory}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        with open(os.path.join(output_directory, "result.json"), "w") as fp:
            results = {"trials": trials, "config": conf, "features_detail": {"number": len(feats), "names": feats}}
            json.dump(results, fp, indent=2)

        feature_importance_fname = os.path.join(output_directory, f'feature_importance.pkl')
        feature_importance_df.to_pickle(feature_importance_fname)

        del feature_importance_df
        gc.collect()

        test['TARGET'] = sub_preds
        test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)
