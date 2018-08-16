import os
import json
from attrdict import AttrDict

from models import LightGBM
from features.application import ApplicationFeatures
from features.bureau import BureauFeatures
from features.previous_application import PreviousApplicationFeatures
from features.pos_cash import PosCashFeatures
from features.installments_payments import InstallmentsPaymentsFeatures
from features.credit_card_balance import CreditCardBalanceFeatures


KEY_FEATURE_MAP = {
    "ApplicationFeatures": ApplicationFeatures,
    "BureauFeatures": BureauFeatures,
    "PreviousApplicationFeatures": PreviousApplicationFeatures,
    "PosCashFeatures": PosCashFeatures,
    "InstallmentsPaymentsFeatures": InstallmentsPaymentsFeatures,
    "CreditCardBalanceFeatures": CreditCardBalanceFeatures
}


def read_config(config_file_path: str) -> dict:
    with open(config_file_path, mode='r', encoding='utf-8') as fp:
        json_txt = fp.read()
        json_txt = str(json_txt).replace("'", '"').replace('True', 'true').replace('False', 'false')
        json_data = AttrDict(json.loads(json_txt))

    json_data.config_file_name = os.path.splitext(os.path.basename(config_file_path))[0]
    return json_data
