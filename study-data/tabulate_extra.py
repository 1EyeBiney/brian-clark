"""Third-stage tabulation, for the "check our work" section of the study.

Usage:
    python tabulate_extra.py <year> <path to csv_pus.zip>

Same universe as the other programs: ages 18 to 64, civilian, not living in an
institution.

Output, in the results folder beside this script:
    wkl_<year>.csv     weighted totals (main weight and 80 replicate weights) by
                       vision difficulty, other difficulty, bachelor's degree,
                       age group, employment status, and when the person last
                       worked (WKL: 1 = within the past 12 months, 2 = 1 to 5
                       years ago, 3 = over 5 years ago or never worked).
    earnhist_<year>.csv  weighted totals for every distinct value of total
                       earnings (PERNP) among people at work, by vision
                       difficulty and place of work, so that medians and their
                       margins of error can be computed from replicate weights.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

REPS = [f'PWGTP{i}' for i in range(1, 81)]
WEIGHTS = ['PWGTP'] + REPS
BASE = ['AGEP', 'DEYE', 'DEAR', 'DREM', 'DPHY', 'DDRS', 'DOUT', 'ESR', 'JWTRNS', 'RELSHIPP', 'SCHL', 'WKL', 'PERNP']


def main():
    year, zpath = sys.argv[1], sys.argv[2]
    wkl = None
    earn = None
    wanted = set(BASE + WEIGHTS)
    with zipfile.ZipFile(zpath) as z:
        for name in [n for n in z.namelist() if n.lower().endswith('.csv')]:
            with z.open(name) as f:
                reader = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), usecols=lambda c: c in wanted,
                                     chunksize=300_000, low_memory=False)
                for chunk in reader:
                    c = chunk[(chunk['AGEP'] >= 18) & (chunk['AGEP'] <= 64) & (chunk['RELSHIPP'] != 37)
                              & chunk['ESR'].notna() & ~chunk['ESR'].isin([4, 5])].copy()
                    c['vision'] = (c['DEYE'] == 1).astype(int)
                    c['other_dis'] = ((c['DEAR'] == 1) | (c['DREM'] == 1) | (c['DPHY'] == 1)
                                      | (c['DDRS'] == 1) | (c['DOUT'] == 1)).astype(int)
                    c['status'] = np.where(c['ESR'].isin([1, 2]), 'employed', np.where(c['ESR'] == 3, 'unemployed', 'nilf'))
                    c['college'] = (c['SCHL'] >= 21).astype(int)
                    c['age'] = np.where(c['AGEP'] <= 34, '18-34', np.where(c['AGEP'] <= 49, '35-49', '50-64'))
                    c['wkl'] = pd.to_numeric(c['WKL'], errors='coerce').fillna(0).astype(int)
                    c['records'] = 1
                    part = c.groupby(['vision', 'other_dis', 'college', 'age', 'status', 'wkl'])[WEIGHTS + ['records']].sum()
                    wkl = part if wkl is None else wkl.add(part, fill_value=0)
                    w = c[c['JWTRNS'].notna()].copy()
                    w['place'] = np.where(w['JWTRNS'] == 11, 'home', 'onsite')
                    w['PERNP'] = pd.to_numeric(w['PERNP'], errors='coerce')
                    part = w.groupby(['vision', 'place', 'PERNP'], dropna=False)[WEIGHTS + ['records']].sum()
                    earn = part if earn is None else earn.add(part, fill_value=0)
    here = os.path.dirname(os.path.abspath(__file__))
    wkl = wkl.reset_index()
    wkl.insert(0, 'year', int(year))
    wkl.to_csv(os.path.join(here, 'results', f'wkl_{year}.csv'), index=False)
    earn = earn.reset_index()
    earn.insert(0, 'year', int(year))
    earn.to_csv(os.path.join(here, 'results', f'earnhist_{year}.csv'), index=False)
    v = wkl[wkl['vision'] == 1]
    print(f"{year}: vision-difficulty employment {100 * v[v['status'] == 'employed']['PWGTP'].sum() / v['PWGTP'].sum():.2f}% "
          f"(check against stage one); wkl rows {len(wkl)}, earnings rows {len(earn)}")


if __name__ == '__main__':
    main()
