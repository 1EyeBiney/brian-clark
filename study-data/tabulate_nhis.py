"""Employment by severity of vision difficulty, from the National Health
Interview Survey (NHIS) Sample Adult public-use files.

    python tabulate_nhis.py            writes results/nhis.json

Files: data_nhis/adultYY.zip for 2019 and 2021 through 2024, from
https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/ . The 2020 file
is left out: part of that year's sample is a re-interview of 2019 respondents,
and the Census study on this page has no 2020 either.

Universe: sample adults ages 18 to 64, the five years pooled.

Vision (VISIONDF_A): "Do you have difficulty seeing, even when wearing glasses
or contact lenses?"  1 no difficulty, 2 some difficulty, 3 a lot of
difficulty, 4 cannot do at all.

Working: EMPWRKLSWK_A (2019) or EMPWRKLSW1_A (2021 on) = 1. This is the
survey's own summary of its work questions. It counts people who worked for pay
last week, people who had a job but were temporarily away from it, and people
working without pay in a family business, so it is close to "employed" in the
Census survey. The survey's documentation says the 2019 version is not exactly
comparable with the later one, which treats seasonal and contract workers
differently. The plain question "did you work for pay last week" is on the
files only from 2022, and runs 2 to 3.5 points lower.

Margins of error are 90 percent, by Taylor linearization for a ratio, using
the public-use strata (PSTRAT) and sampling units (PPSU), with strata kept
separate for each year.
"""
import json
import os
import zipfile

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
YEARS = [2019, 2021, 2022, 2023, 2024]
OTHER = ['HEARINGDF_A', 'DIFF_A', 'COGMEMDFF_A', 'UPPSLFCR_A', 'COMDIFF_A']
frames = []
for y in YEARS:
    yy = str(y)[2:]
    with zipfile.ZipFile(os.path.join(here, 'data_nhis', f'adult{yy}.zip')) as z:
        d = pd.read_csv(z.open(f'adult{yy}.csv'), low_memory=False)
    work = 'EMPWRKLSWK_A' if 'EMPWRKLSWK_A' in d.columns else 'EMPWRKLSW1_A'
    edu = 'EDUC_A' if 'EDUC_A' in d.columns else 'EDUCP_A'
    d = d[(d['AGEP_A'] >= 18) & (d['AGEP_A'] <= 64)]
    f = pd.DataFrame({'year': y, 'w': d['WTFA_A'].astype(float), 'strata': y * 1000 + d['PSTRAT'], 'psu': d['PPSU'],
                      'vision': d['VISIONDF_A'], 'work': d[work], 'edu': d[edu], 'age': d['AGEP_A'],
                      'other': (d[OTHER].isin([3, 4])).any(axis=1).astype(int),
                      'other_known': (d[OTHER].isin([1, 2, 3, 4])).all(axis=1)})
    frames.append(f)
df = pd.concat(frames, ignore_index=True)
df['worked'] = (df['work'] == 1).astype(float)
df['work_known'] = df['work'].isin([1, 2])
df['college'] = np.where(df['edu'].between(8, 11), 1, np.where(df['edu'].between(0, 7), 0, -1))
df['agegrp'] = np.where(df['age'] <= 34, '18-34', np.where(df['age'] <= 49, '35-49', '50-64'))


def ratio(mask_den, num_col='worked'):
    """Weighted percent with Taylor-linearized 90% margin of error. The whole sample stays in the variance calculation."""
    m = mask_den.to_numpy()
    w = df['w'].to_numpy()
    x = np.where(m, w, 0.0)
    y = x * df[num_col].to_numpy()
    X, Y = x.sum(), y.sum()
    if X == 0:
        return None
    R = Y / X
    z = (y - R * x) / X
    t = pd.DataFrame({'s': df['strata'].to_numpy(), 'p': df['psu'].to_numpy(), 'z': z}).groupby(['s', 'p'])['z'].sum().reset_index()
    var = 0.0
    for _, g in t.groupby('s'):
        n = len(g)
        if n > 1:
            var += n / (n - 1.0) * float(((g['z'] - g['z'].mean()) ** 2).sum())
    se = var ** 0.5
    return {'pct': round(float(100 * R), 1), 'moe': round(float(100 * 1.645 * se), 1), 'se': round(float(100 * se), 3),
            'pct_exact': float(100 * R), 'moe_exact': float(100 * 1.645 * se),
            'records': int(m.sum()), 'people_per_year': int(round(X / len(YEARS)))}


GROUPS = {'none': [1], 'some': [2], 'a_lot': [3], 'cannot': [4], 'a_lot_or_cannot': [3, 4], 'any': [2, 3, 4]}
out = {'years': YEARS, 'groups': {}}
known = df['work_known']
for name, codes in GROUPS.items():
    v = df['vision'].isin(codes)
    g = {'all': ratio(v & known)}
    g['records_all'] = int(v.sum())
    g['people_per_year_all'] = int(round(df.loc[v, 'w'].sum() / len(YEARS)))
    for col in (1, 0):
        g[f'college{col}'] = ratio(v & known & (df['college'] == col))
    for od in (1, 0):
        g[f'other{od}'] = ratio(v & known & (df['other'] == 1) if od else v & known & df['other_known'] & (df['other'] == 0))
    for a in ('18-34', '35-49', '50-64'):
        g[f'age_{a}'] = ratio(v & known & (df['agegrp'] == a))
    for y in YEARS:
        g[f'year_{y}'] = ratio(v & known & (df['year'] == y))
    out['groups'][name] = g
    print(name, g['records_all'], g['people_per_year_all'], g['all'])
    for k in ('college1', 'college0', 'other1', 'other0', 'age_18-34', 'age_35-49', 'age_50-64'):
        print('   ', k, g[k])
out['everyone'] = {str(y): ratio(known & (df['year'] == y)) for y in YEARS}
out['everyone']['pooled'] = ratio(known)
print('everyone', out['everyone'])
tot = df['vision'].isin([1, 2, 3, 4])
out['share_of_adults'] = {name: round(float(100) * df.loc[df['vision'].isin(codes), 'w'].sum() / df.loc[tot, 'w'].sum(), 2) for name, codes in GROUPS.items()}
print('share of adults 18-64', out['share_of_adults'])
with open(os.path.join(here, 'results', 'nhis.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
