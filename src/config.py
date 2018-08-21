import os
import json
from attrdict import AttrDict

from models import LightGBM, XGBoost, LogReg, RandomForest
from features.application import ApplicationFeatures, ApplicationFeaturesOpenSolution, ApplicationFeaturesLeakyTargetEncoding
from features.bureau import BureauFeatures, BureauFeaturesOpenSolution
from features.previous_application import PreviousApplicationFeatures, PreviousApplicationFeaturesOpenSolution
from features.pos_cash import PosCashFeatures, PosCashFeaturesOpenSolution
from features.installments_payments import InstallmentsPaymentsFeatures, InstallmentsPaymentsFeaturesOpenSolution
from features.credit_card_balance import CreditCardBalanceFeatures, CreditCardBalanceFeaturesOpenSolution
from features.agg_application import AggregateFeatureApplicationOpenSolution
from features.agg_bureau import AggregateFeatureBureauOpenSolution
from features.agg_credit_card_balance import AggregateFeatureCreditCardBalanceOpenSolution
from features.agg_installments_payments import AggregateFeatureInstallmentsPaymentsOpenSolution
from features.agg_pos_cash import AggregateFeaturePosCashOpenSolution
from features.agg_previous_application import AggregateFeaturePreviousApplicationOpenSolution
from features.stacking import LGBM1_1, LogReg0


KEY_FEATURE_MAP = {
    "ApplicationFeatures": ApplicationFeatures,
    "ApplicationFeaturesOpenSolution": ApplicationFeaturesOpenSolution,
    "BureauFeatures": BureauFeatures,
    "BureauFeaturesOpenSolution": BureauFeaturesOpenSolution,
    "PreviousApplicationFeatures": PreviousApplicationFeatures,
    "PreviousApplicationFeaturesOpenSolution": PreviousApplicationFeaturesOpenSolution,
    "PosCashFeatures": PosCashFeatures,
    "PosCashFeaturesOpenSolution": PosCashFeaturesOpenSolution,
    "InstallmentsPaymentsFeatures": InstallmentsPaymentsFeatures,
    "InstallmentsPaymentsFeaturesOpenSolution": InstallmentsPaymentsFeaturesOpenSolution,
    "CreditCardBalanceFeatures": CreditCardBalanceFeatures,
    "CreditCardBalanceFeaturesOpenSolution": CreditCardBalanceFeaturesOpenSolution,
    "AggregateFeatureApplicationOpenSolution": AggregateFeatureApplicationOpenSolution,
    "AggregateFeatureBureauOpenSolution": AggregateFeatureBureauOpenSolution,
    "AggregateFeatureCreditCardBalanceOpenSolution": AggregateFeatureCreditCardBalanceOpenSolution,
    "AggregateFeatureInstallmentsPaymentsOpenSolution": AggregateFeatureInstallmentsPaymentsOpenSolution,
    "AggregateFeaturePosCashOpenSolution": AggregateFeaturePosCashOpenSolution,
    "AggregateFeaturePreviousApplicationOpenSolution": AggregateFeaturePreviousApplicationOpenSolution,
    "ApplicationFeaturesLeakyTargetEncoding": ApplicationFeaturesLeakyTargetEncoding,
    "PredsLGBM1_1": LGBM1_1,
    "PredsLogReg0": LogReg0
}


KEY_MODEL_MAP = {
    "LightGBM": LightGBM,
    "XGBoost": XGBoost,
    "LogisticRegression": LogReg,
    "RandomForest": RandomForest
}


def read_config(config_file_path: str) -> dict:
    with open(config_file_path, mode='r', encoding='utf-8') as fp:
        json_txt = fp.read()
        json_txt = str(json_txt).replace("'", '"').replace('True', 'true').replace('False', 'false')
        config = AttrDict(json.loads(json_txt))

    # set dafault values
    config.config_file_name = os.path.splitext(os.path.basename(config_file_path))[0]
    if "name" not in config.model:
        config.model = {**config.model, "name": "LightGBM"}
    if "options" not in config:
        config.options = {"drop_duplicate_column_on_merge": False}

    return config
