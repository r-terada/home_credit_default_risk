import os
import gc
import numpy as np
import pandas as pd
from typing import Optional


class RawData:
    _df: Optional[pd.DataFrame] = None
    _file_name: str = ""

    @classmethod
    def get_df(cls, conf) -> pd.DataFrame:
        if cls._df is None:
            cls._df = cls._read_data(conf)

        return cls._df

    @classmethod
    def _read_data(cls, conf) -> pd.DataFrame:
        pickled_fname = os.path.join(conf.dataset.cache_directory, 'raw_data', f'{cls._file_name}.pkl')
        if os.path.exists(pickled_fname):
            return pd.read_pickle(pickled_fname)
        else:
            print('read data from csv')
            df = cls._read_csv(conf)  # pd.DataFrame
            print('clean dataframe.')
            df = cls._clean_feature(conf, df)
            print(f'save to {pickled_fname} as pickled file')
            df.to_pickle(pickled_fname)
            return df

    @classmethod
    def _read_csv(cls, conf) -> pd.DataFrame:
        return pd.read_csv(os.path.join(conf.dataset.input_directory, f'{cls._file_name}.csv'))

    @classmethod
    def _clean_feature(cls, conf, df) -> pd.DataFrame:
        return df


class Application(RawData):
    _file_name = "application"

    # override (need to join train and test csv at first time)
    @classmethod
    def _read_csv(cls, conf) -> pd.DataFrame:
        train_df = pd.read_csv(os.path.join(conf.dataset.input_directory, 'application_train.csv'))  # pd.DataFrame
        test_df = pd.read_csv(os.path.join(conf.dataset.input_directory, 'application_test.csv'))  # pd.DataFrame
        df = train_df.append(test_df, sort=True).reset_index()
        del train_df, test_df
        gc.collect()
        return df

    # override
    @classmethod
    def _clean_feature(cls, conf, df) -> pd.DataFrame:
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
        return df


class Bureau(RawData):
    _file_name = "bureau"

    # override
    @classmethod
    def _clean_feature(cls, conf, df) -> pd.DataFrame:
        df.loc[df['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan
        df.loc[df['DAYS_CREDIT_UPDATE'] < -40000, 'DAYS_CREDIT_UPDATE'] = np.nan
        df.loc[df['DAYS_ENDDATE_FACT'] < -40000, 'DAYS_ENDDATE_FACT'] = np.nan
        return df


class BureauBalance(RawData):
    _file_name = "bureau_balance"


class CreditCardBalance(RawData):
    _file_name = "credit_card_balance"

    # override
    def _clean_feature(cls, conf, df) -> pd.DataFrame:
        df.loc[df['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
        df.loc[df['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan
        return df


class PosCash(RawData):
    _file_name = "POS_CASH_balance"


class PreviousApplication(RawData):
    _file_name = "previous_application"

    # override
    def _clean_feature(cls, conf, df) -> pd.DataFrame:
        df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
        df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        return df


class InstallmentsPayments(RawData):
    _file_name = "installments_payments"


def main():
    import time
    from contextlib import contextmanager

    @contextmanager
    def timer(title):
        print(f"== {title}")
        t0 = time.time()
        yield
        print("== done. {:.0f} [s]".format(time.time() - t0))

    from attrdict import AttrDict
    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../data"
    )
    conf = AttrDict({
        "dataset": {
            "input_directory": os.path.join(DATA_DIR, "input"),
            "cache_directory": os.path.join(DATA_DIR, "working")
        }
    })

    for _cls in [Application,
                 Bureau,
                 BureauBalance,
                 CreditCardBalance,
                 PosCash,
                 PreviousApplication,
                 InstallmentsPayments]:
        with timer(f"reading {_cls.__name__}"):
            df = _cls.get_df(conf)
            print(df.head())
            del df
            gc.collect


if __name__ == '__main__':
    main()
