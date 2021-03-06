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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


class Model:

    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> float:
        raise NotImplementedError


class LightGBM(Model):

    # override
    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> float:
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
            "val_score_mean": np.mean([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)]),
            "val_score_std": np.std([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)]),
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

        oof_preds_df = pd.DataFrame()
        oof_preds_df['SK_ID_CURR'] = train['SK_ID_CURR']
        oof_preds_df['TARGET'] = oof_preds
        oof_preds_df.to_csv(os.path.join(output_directory, f'oof_predictions.csv'), index=False)
        del oof_preds_df
        gc.collect()

        test['TARGET'] = sub_preds
        test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

        return score


class XGBoost(Model):

    # override
    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict
                                ) -> float:
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
            "val_score_mean": np.mean([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)]),
            "val_score_std": np.std([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)]),
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

        oof_preds_df = pd.DataFrame()
        oof_preds_df['SK_ID_CURR'] = train['SK_ID_CURR']
        oof_preds_df['TARGET'] = oof_preds
        oof_preds_df.to_csv(os.path.join(output_directory, f'oof_predictions.csv'), index=False)
        del oof_preds_df
        gc.collect()

        test['TARGET'] = sub_preds
        test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

        return score


class SKLearnClassifier(Model):

    def _get_clf_class(self):
        return None

    def train_and_predict(self,
                          train: pd.DataFrame,
                          test: pd.DataFrame,
                          feats: List[str],
                          target: str,
                          conf: AttrDict,
                          ) -> None:
        x_train = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train[target]
        x_test = test[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        if "normalize_feature" in conf and conf.normalize_feature:
            sc = StandardScaler()
            train_x, train_y = sc.fit_transform(x_train, y_train), y_train
            test_x = sc.transform(x_test)
        else:
            train_x, train_y = x_train, y_train
            test_x = x_test

        clf = self._get_clf_class()(**conf.model.clf_params)
        clf.fit(train_x, train_y, **conf.model.train_params)

        sub_preds = clf.predict_proba(test_x)[:, 1]

        # write results
        output_directory = os.path.join(
            conf.dataset.output_directory,
            f"{datetime.now().strftime('%m%d%H%M')}_{conf.config_file_name}_train_full_data"
        )
        print(f"write results to {output_directory}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        test['TARGET'] = sub_preds
        test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

    # override
    def train_and_predict_kfold(self,
                                train: pd.DataFrame,
                                test: pd.DataFrame,
                                feats: List[str],
                                target: str,
                                conf: AttrDict,
                                save_result: bool=True,
                                verbose: bool=True
                                ) -> float:
        # prepare
        x_train = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train[target]
        x_test = test[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        sub_preds = np.zeros(x_test.shape[0])
        oof_preds = np.zeros(x_train.shape[0])
        trials = {}

        # cv
        folds = StratifiedKFold(**conf.model.kfold_params)
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(x_train, y_train)):
            start_time = time.time()
            if "normalize_feature" in conf and conf.normalize_feature:
                sc = StandardScaler()
                train_x, train_y = sc.fit_transform(x_train.iloc[train_idx].copy()), y_train.iloc[train_idx].copy()
                valid_x, valid_y = sc.transform(x_train.iloc[valid_idx].copy()), y_train.iloc[valid_idx].copy()
            else:
                train_x, train_y = x_train.iloc[train_idx].copy(), y_train.iloc[train_idx].copy()
                valid_x, valid_y = x_train.iloc[valid_idx].copy(), y_train.iloc[valid_idx].copy()

            clf = self._get_clf_class()(**conf.model.clf_params)
            clf.fit(train_x, train_y, **conf.model.train_params)

            oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            if "normalize_feature" in conf and conf.normalize_feature:
                sub_preds += clf.predict_proba(sc.transform(x_test))[:, 1] / folds.n_splits
            else:
                sub_preds += clf.predict_proba(x_test)[:, 1] / folds.n_splits

            score = roc_auc_score(valid_y, oof_preds[valid_idx])
            if verbose:
                print(f'Fold {(n_fold + 1):2d} AUC : {score:.6f}')

            trials[f"Fold{n_fold + 1}"] = {
                "val_score": score,
                "elapsed_time": time.time() - start_time
            }

            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()

        score = roc_auc_score(y_train, oof_preds)
        if verbose:
            print(f'Full AUC score {score:.6f}')

        if save_result:
            trials["Full"] = {
                "score": score,
                "val_score_mean": np.mean([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)]),
                "val_score_std": np.std([trials[f"Fold{n_fold + 1}"]["val_score"] for n_fold in range(folds.n_splits)])
           }

            # write results
            output_directory = os.path.join(
                conf.dataset.output_directory,
                f"{datetime.now().strftime('%m%d%H%M')}_{conf.config_file_name}_{score:.6f}"
            )
            if verbose:
                print(f"write results to {output_directory}")
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            with open(os.path.join(output_directory, "result.json"), "w") as fp:
                results = {"trials": trials, "config": conf, "features_detail": {"number": len(feats), "names": feats}}
                json.dump(results, fp, indent=2)

            oof_preds_df = pd.DataFrame()
            oof_preds_df['SK_ID_CURR'] = train['SK_ID_CURR']
            oof_preds_df['TARGET'] = oof_preds
            oof_preds_df.to_csv(os.path.join(output_directory, f'oof_predictions.csv'), index=False)
            del oof_preds_df
            gc.collect()

            test['TARGET'] = sub_preds
            test[['SK_ID_CURR', 'TARGET']].to_csv(os.path.join(output_directory, 'submission.csv'), index=False)

        return score


class LogReg(SKLearnClassifier):

    def _get_clf_class(self):
        return LogisticRegression


class RandomForest(SKLearnClassifier):

    def _get_clf_class(self):
        return RandomForestClassifier


class KNN(SKLearnClassifier):

    def _get_clf_class(self):
        return KNeighborsClassifier
