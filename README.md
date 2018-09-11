# home_credit_default_risk
21st place solution of https://www.kaggle.com/c/home-credit-default-risk

## usage

1. download and extract data
```
$ kaggle competitions download --path data/input home-credit-default-risk
$ cd data/input
$ unzip *
```

2. run
```
$ cd src
$ python run.py --config_file configs/lgbm_0.json
```
