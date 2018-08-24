import os
import click
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 20)


def simple_blend(result_files):
    dfs = [pd.read_csv(f) for f in result_files]
    sub = pd.DataFrame({'SK_ID_CURR': dfs[0].SK_ID_CURR.unique()})
    sub['TARGET'] = np.mean([df.TARGET for df in dfs], axis=0)
    return sub


def rank_average(result_files):
    data = {}
    for f in result_files:
        data['/'.join(f.strip().split('/')[-2:])] = pd.read_csv(f)
    ranks = pd.DataFrame(columns=data.keys())
    for key in data.keys():
        ranks[key] = data[key].TARGET.rank(method='min')
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    print(ranks.corr())
    weights = [0.05, 0.2, 0.3, 0.05, 0.05, 0.2, 0.05, 0.05, 0.05]
    print(list(data.keys()))
    ranks['Score'] = ranks[list(data.keys())].mul(weights).sum(1) / ranks.shape[0]
    sub = pd.DataFrame({'SK_ID_CURR': list(data.values())[0].SK_ID_CURR.unique()})
    sub['TARGET'] = ranks['Score']
    return sub


@click.command()
@click.option('--out_file', type=str, default='./blend.csv')
@click.option('--method', type=str, default='blend')
def main(out_file, method):
    result_dirs = [os.path.join(
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/output/', fname, 'submission.csv'
    ) for fname in [
        '08192245_logreg_0_0.777166',
        '08181653_xgb_0_0.793305',
        '08230034_lgbm_3-3_seed0_0.796089'
    ]] + [os.path.join(
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/ext_output/', fname
    ) for fname in [
        'catboost.787.csv',
        'gpalldatasubmission.785.csv',
        'new1submit8aug.802.csv',
        'stack3_diff_data.789.csv',
        'sub_nn.775.csv',
        'tidy_xgb_continuous_features_0.79031.790.csv'
    ]]
    if method == 'blend':
        sub = simple_blend(result_dirs)
    elif method == 'rank_average':
        sub = rank_average(result_dirs)
    sub.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
