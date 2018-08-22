import os
import gc
import numpy as np
import pandas as pd

from collections import defaultdict
from category_encoders import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from features import Feature
from features.raw_data import Application


def create_word_list(df: pd.DataFrame, col1: str, col2: str):
    col1_size = df[col1].max() + 1
    col2_list = [[] for _ in range(col1_size)]
    for val1, val2 in zip(df[col1], df[col2]):
        col2_list[val1].append(val2)
    return [' '.join(map(str, list)) for list in col2_list]


def lda(n_components, df, col1, col2):
    word_list = create_word_list(df, col1, col2)
    document_term = CountVectorizer().fit_transform(word_list)
    latent_vectors = LatentDirichletAllocation(n_components, learning_method='online', random_state=71).fit_transform(document_term)
    return latent_vectors


class ApplicationFeaturesLDA(Feature):
    _col1 = ""
    _col2 = ""
    _n_components = 5

    @classmethod
    def _create_feature(cls, conf) -> pd.DataFrame:
        df = Application.get_df(conf)[['SK_ID_CURR', 'TARGET', cls._col1, cls._col2]]
        df = OrdinalEncoder(cols=[cls._col1, cls._col2]).fit_transform(df)
        latent_vectors = lda(cls._n_components, df, cls._col1, cls._col2)

        dic = defaultdict(list)
        for v in latent_vectors:
            for i, s in enumerate(v):
                dic[f"{cls._col1}_LDA_{cls._col2}_dim{i}"].append(s)
        df_latent_vectors = pd.DataFrame(dic)
        return df.merge(df_latent_vectors, how="left", left_on=cls._col1, right_index=True).drop(['TARGET', cls._col1, cls._col2], axis=1)


class ApplicationFeaturesLDAOccupationTypeOrganizationType5(ApplicationFeaturesLDA):
    _col1 = "OCCUPATION_TYPE"
    _col2 = "ORGANIZATION_TYPE"
    _n_components = 5


class ApplicationFeaturesLDAOrganizationTypeOccupationType5(ApplicationFeaturesLDA):
    _col1 = "ORGANIZATION_TYPE"
    _col2 = "OCCUPATION_TYPE"
    _n_components = 5
