"""Second-stage tabulation: build a compact summary "cube" from one year of the
Census Bureau's ACS public microdata, for the follow-up questions in the study.

Usage:
    python tabulate_detail.py <year> <path to csv_pus.zip or a state file>

Universe: ages 18 to 64, civilian, not living in an institution (same as
tabulate_acs.py).

Output, in the results folder beside this script:
    cube_<year>_<file>.csv      weighted totals (main weight and 80 replicate
                                weights) for every combination of the categories
                                below. Every table in the study is a sum over
                                rows of this file.
    earn_<year>_<file>.json     weighted median earnings and hours for people at
                                work, by vision difficulty and place of work.

Categories:
    indiana     1 if the person lives in Indiana (state code 18)
    vision      1 if vision difficulty (DEYE = 1)
    other_dis   1 if any other difficulty is also reported (hearing, cognitive,
                walking, self-care, independent living)
    status      employed / unemployed / nilf (not in labor force)
    place       home / onsite / not_at_work   (JWTRNS: 11 = worked from home)
    college     1 if bachelor's degree or higher (SCHL >= 21)
    occ         occupation group of employed people, from OCCP
    age         18-34 / 35-49 / 50-64
    selfemp     1 if self-employed (COW 6 or 7)
    fulltime    1 if usually works 35 or more hours a week
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

REPS = [f'PWGTP{i}' for i in range(1, 81)]
WEIGHTS = ['PWGTP'] + REPS
BASE = ['AGEP', 'DEYE', 'DEAR', 'DREM', 'DPHY', 'DDRS', 'DOUT', 'ESR', 'JWTRNS', 'RELSHIPP',
        'SCHL', 'OCCP', 'COW', 'WKHP', 'PERNP']
STATE_NAMES = ('ST', 'STATE')
KEYS = ['indiana', 'vision', 'other_dis', 'status', 'place', 'college', 'occ', 'age', 'selfemp', 'fulltime']


def occ_group(code):
    """Occupation groups from 2018 Census occupation codes."""
    out = np.full(len(code), 'none', dtype=object)
    c = code.to_numpy(dtype='float64')
    out[(c >= 10) & (c <= 3550)] = 'management_professional'
    out[(c >= 3601) & (c <= 4655)] = 'service'
    out[(c >= 4700) & (c <= 4965)] = 'sales'
    out[(c >= 5000) & (c <= 5940)] = 'office_admin'
    out[(c >= 6005) & (c <= 9760)] = 'construction_production_transport'
    return out


def weighted_median(values, weights):
    if len(values) == 0:
        return None
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


def main():
    year, zpath = sys.argv[1], sys.argv[2]
    cube = None
    earn_rows = []
    with zipfile.ZipFile(zpath) as z:
        for name in [n for n in z.namelist() if n.lower().endswith('.csv')]:
            with z.open(name) as f:
                wanted = set(BASE + WEIGHTS + list(STATE_NAMES))
                reader = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), usecols=lambda c: c in wanted,
                                     chunksize=300_000, low_memory=False)
                for chunk in reader:
                    st_col = 'ST' if 'ST' in chunk.columns else 'STATE'
                    c = chunk[(chunk['AGEP'] >= 18) & (chunk['AGEP'] <= 64) & (chunk['RELSHIPP'] != 37)
                              & chunk['ESR'].notna() & ~chunk['ESR'].isin([4, 5])].copy()
                    c['indiana'] = (pd.to_numeric(c[st_col], errors='coerce') == 18).astype(int)
                    c['vision'] = (c['DEYE'] == 1).astype(int)
                    c['other_dis'] = ((c['DEAR'] == 1) | (c['DREM'] == 1) | (c['DPHY'] == 1)
                                      | (c['DDRS'] == 1) | (c['DOUT'] == 1)).astype(int)
                    c['status'] = np.where(c['ESR'].isin([1, 2]), 'employed', np.where(c['ESR'] == 3, 'unemployed', 'nilf'))
                    c['place'] = np.where(c['JWTRNS'] == 11, 'home', np.where(c['JWTRNS'].notna(), 'onsite', 'not_at_work'))
                    c['college'] = (c['SCHL'] >= 21).astype(int)
                    c['occ'] = np.where(c['status'] == 'employed', occ_group(pd.to_numeric(c['OCCP'], errors='coerce')), 'none')
                    c['age'] = np.where(c['AGEP'] <= 34, '18-34', np.where(c['AGEP'] <= 49, '35-49', '50-64'))
                    c['selfemp'] = c['COW'].isin([6, 7]).astype(int)
                    c['fulltime'] = (c['WKHP'] >= 35).astype(int)
                    c['records'] = 1
                    part = c.groupby(KEYS)[WEIGHTS + ['records']].sum()
                    cube = part if cube is None else cube.add(part, fill_value=0)
                    w = c[c['place'] != 'not_at_work'][['vision', 'place', 'PERNP', 'WKHP', 'PWGTP']]
                    earn_rows.append(w)

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    tag = os.path.basename(zpath).replace('.zip', '')
    cube = cube.reset_index()
    cube.insert(0, 'year', int(year))
    dest = os.path.join(here, 'results', f'cube_{year}_{tag}.csv')
    cube.to_csv(dest, index=False)

    e = pd.concat(earn_rows)
    earn = {'year': int(year), 'note': 'weighted medians, people at work, nominal dollars for that survey year'}
    for vis in (1, 0):
        for place in ('home', 'onsite'):
            sub = e[(e['vision'] == vis) & (e['place'] == place)]
            earn[f"{'vision' if vis else 'no_vision'}_{place}"] = {
                'median_earnings': weighted_median(sub['PERNP'].to_numpy(dtype='float64'), sub['PWGTP'].to_numpy(dtype='float64')),
                'median_hours': weighted_median(sub['WKHP'].fillna(0).to_numpy(dtype='float64'), sub['PWGTP'].to_numpy(dtype='float64')),
                'sample_records': int(len(sub)),
            }
    with open(os.path.join(here, 'results', f'earn_{year}_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(earn, f, indent=2)

    v = cube[cube['vision'] == 1]
    emp = v[v['status'] == 'employed']['PWGTP'].sum() / v['PWGTP'].sum() * 100
    print(f'{year}: cube rows {len(cube)}, vision-difficulty employment {emp:.2f}% (check against stage one). Saved {dest}')


if __name__ == '__main__':
    main()
