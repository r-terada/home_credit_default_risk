import os
import click
import numpy as np
import pandas as pd


def simple_blend(result_dirs):
    dfs = [pd.read_csv(os.path.join(d, "submission.csv")) for d in result_dirs]
    sub = pd.DataFrame({'SK_ID_CURR': dfs[0].SK_ID_CURR.unique()})
    sub['TARGET'] = np.mean([df.TARGET for df in dfs], axis=0)
    return sub


def rank_average(result_dirs):
    data = {}
    for d in result_dirs:
        data[os.path.basename(d)] = pd.read_csv(os.path.join(d, "submission.csv"))
    ranks = pd.DataFrame(columns=data.keys())
    for key in data.keys():
        ranks[key] = data[key].TARGET.rank(method='min')
    ranks['Average'] = ranks.mean(axis=1)
    ranks['Scaled Rank'] = (ranks['Average'] - ranks['Average'].min()) / (ranks['Average'].max() - ranks['Average'].min())
    print(ranks.corr())
    # weights = [1 / len(data.keys()) for _ in range(len(data.keys()))]
    weights = [0.45, 0.45, 0.05, 0.05]
    print(list(data.keys()))
    ranks['Score'] = ranks[list(data.keys())].mul(weights).sum(1) / ranks.shape[0]
    sub = pd.DataFrame({'SK_ID_CURR': list(data.values())[0].SK_ID_CURR.unique()})
    sub['TARGET'] = ranks['Score']
    return sub


@click.command()
@click.option('--out_file', type=str, default='./blend.csv')
@click.option('--method', type=str, default='blend')
def main(out_file, method):
    result_dirs = [
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/output/08230034_lgbm_3-3_seed0_0.796089',
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/output/08211432_lgbm_3-1-0_0.795889',
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/output/08192245_logreg_0_0.777166',
        '/Users/rintaro/.ghq/github.com/r-terada/home_credit_default_risk/data/output/08181653_xgb_0_0.793305'
    ]
    if method == 'blend':
        sub = simple_blend(result_dirs)
    elif method == 'rank_average':
        sub = rank_average(result_dirs)
    sub.to_csv(out_file, index=False)


if __name__ == '__main__':
    main()
