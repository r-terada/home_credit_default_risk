import os
import glob
import pandas as pd
from functools import reduce

from features import Feature


class StackingFeature(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        dir_name = os.path.join(conf.dataset.output_directory, cls._name)
        train_predictions = pd.read_csv(os.path.join(dir_name, "oof_predictions.csv"))
        test_predictions = pd.read_csv(os.path.join(dir_name, "submission.csv"))
        df = train_predictions.append(test_predictions, sort=True).reset_index()
        df = df.rename({"TARGET": f"{cls.__name__}_PREDICTION"}, axis="columns")
        return df


class LGBM1_1(StackingFeature):
    _name = "08191504_lgbm_1-1_0.793821"


class LogReg0(StackingFeature):
    _name = "08192245_logreg_0_0.777166"


class LGBM3_1(StackingFeature):
    _name = "08210425_lgbm_3-1_0.794708"


class LGBM3_1_0(StackingFeature):
    _name = "08211432_lgbm_3-1-0_0.795889"


# do not extends Feature class not to persist the feature
class All:

    @classmethod
    def get_df(cls, conf) -> pd.DataFrame:
        dfs = []
        for dir_name in glob.glob(f"{conf.dataset.output_directory}/*"):
            if os.path.basename(dir_name) in ["08192001_logreg_0_0.541404_PREDICTION", "08201837_lgbm_dev_0.534762_PREDICTION"]:
                continue

            try:
                train_predictions = pd.read_csv(os.path.join(dir_name, "oof_predictions.csv"))
                test_predictions = pd.read_csv(os.path.join(dir_name, "submission.csv"))
                df = train_predictions.append(test_predictions, sort=True).reset_index()
                df = df.rename({"TARGET": f"{os.path.basename(dir_name)}_PREDICTION"}, axis="columns")
                dfs.append(df)
            except:
                pass
        ret = reduce(lambda lhs, rhs: lhs.merge(rhs, on=["SK_ID_CURR", "index"], how="left"), dfs)
        print(ret.head())
        return ret
