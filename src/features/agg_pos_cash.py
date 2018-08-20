import os
import sys
import pickle
import pandas as pd
from features.aggs import GroupByAggregateFeature
from features.raw_data import PosCash


POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]


class AggregateFeaturePosCashOpenSolution(GroupByAggregateFeature):
    _base_feature_class = PosCash
    _groupby_aggregations = POS_CASH_BALANCE_AGGREGATION_RECIPIES

    # override
    @classmethod
    def _read_pickle(cls, pkl_fpath) -> pd.DataFrame:
        df = try_to_load_as_pickled_object_or_None(pkl_fpath)
        if df is None:
            print(f"[{cls.__name__}] error in read pickled data")
        return df

    # override
    @classmethod
    def _save_as_pickled_object(cls, df, pkl_fpath) -> None:
        save_as_pickled_object(df, pkl_fpath)


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj