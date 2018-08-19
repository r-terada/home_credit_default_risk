import os
import json
from attrdict import AttrDict

from models import LightGBM, XGBoost, LogReg
from features.application import ApplicationFeatures, ApplicationFeaturesOpenSolution
from features.bureau import BureauFeatures, BureauFeaturesOpenSolution
from features.previous_application import PreviousApplicationFeatures, PreviousApplicationFeaturesOpenSolution
from features.pos_cash import PosCashFeatures, PosCashFeaturesOpenSolution
from features.installments_payments import InstallmentsPaymentsFeatures, InstallmentsPaymentsFeaturesOpenSolution
from features.credit_card_balance import CreditCardBalanceFeatures, CreditCardBalanceFeaturesOpenSolution


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
    "CreditCardBalanceFeaturesOpenSolution": CreditCardBalanceFeaturesOpenSolution
}


KEY_MODEL_MAP = {
    "LightGBM": LightGBM,
    "XGBoost": XGBoost,
    "LogisticRegression": LogReg
}


def read_config(config_file_path: str) -> dict:
    with open(config_file_path, mode='r', encoding='utf-8') as fp:
        json_txt = fp.read()
        json_txt = str(json_txt).replace("'", '"').replace('True', 'true').replace('False', 'false')
        json_data = AttrDict(json.loads(json_txt))

    json_data.config_file_name = os.path.splitext(os.path.basename(config_file_path))[0]
    return json_data
