import pandas as pd
from features import Feature
from features.raw_data import Application


class Base(Feature):

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        return Application.get_df(conf)[["SK_ID_CURR", "TARGET"]]
