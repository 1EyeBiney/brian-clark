"""Tabulate employment and work-from-home by vision difficulty from the Census
Bureau's public American Community Survey microdata (PUMS, 1-year person files).

Usage:
    python tabulate_acs.py <year> <path to csv_pus.zip or a state file such as csv_pak.zip>

Source files: https://www2.census.gov/programs-surveys/acs/data/pums/<year>/1-Year/csv_pus.zip

Universe: ages 18 to 64, civilian, not living in an institution. This matches
the universe the National Research and Training Center on Blindness and Low
Vision uses for its published 2023 figures, so the 2023 result here can be
checked against theirs (52.3 percent employed).

Variables used:
    AGEP      age
    DEYE      vision difficulty: 1 yes, 2 no ("blind or has serious difficulty
              seeing even when wearing glasses")
    ESR       employment status: 1 employed at work, 2 employed not at work,
              3 unemployed, 4 and 5 armed forces, 6 not in labor force
    JWTRNS    means of transportation to work; 11 = worked from home.
              Asked only of people at work in the reference week.
    RELSHIPP  37 = living in an institution
    PWGTP     person weight; PWGTP1..PWGTP80 replicate weights for margins of error

Output: one JSON file per year in the results folder beside this script.
"""
import io
import json
import os
import sys
import zipfile

import pandas as pd

REPS = [f'PWGTP{i}' for i in range(1, 81)]
COLS = ['AGEP', 'DEYE', 'ESR', 'JWTRNS', 'RELSHIPP', 'PWGTP'] + REPS
WEIGHTS = ['PWGTP'] + REPS


def main():
    year, zpath = sys.argv[1], sys.argv[2]
    # sums[group][measure] = array of 81 weighted totals (main weight + 80 replicates)
    sums = {}
    unweighted = {}
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if n.lower().endswith('.csv')]
        for name in members:
            with z.open(name) as f:
                for chunk in pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), usecols=COLS, chunksize=400_000, low_memory=False):
                    c = chunk[(chunk['AGEP'] >= 18) & (chunk['AGEP'] <= 64)]
                    c = c[c['RELSHIPP'] != 37]
                    c = c[~c['ESR'].isin([4, 5])]
                    c = c[c['ESR'].notna()]
                    for group, sub in (('vision', c[c['DEYE'] == 1]), ('no_vision', c[c['DEYE'] == 2])):
                        measures = {
                            'population': sub,
                            'employed': sub[sub['ESR'].isin([1, 2])],
                            'unemployed': sub[sub['ESR'] == 3],
                            'not_in_labor_force': sub[sub['ESR'] == 6],
                            'at_work': sub[sub['JWTRNS'].notna()],
                            'worked_from_home': sub[sub['JWTRNS'] == 11],
                        }
                        for m, rows in measures.items():
                            tot = rows[WEIGHTS].sum().to_numpy(dtype='float64')
                            key = (group, m)
                            sums[key] = sums.get(key, 0) + tot
                            unweighted[key] = unweighted.get(key, 0) + len(rows)

    def ratio(group, num, den, scale=100.0):
        """Estimate, and 90 percent margin of error from the 80 replicate weights."""
        n, d = sums[(group, num)], sums[(group, den)]
        r = scale * n / d
        est = float(r[0])
        se = (4.0 / 80.0 * float(((r[1:] - est) ** 2).sum())) ** 0.5
        return {'estimate': round(est, 2), 'moe90': round(1.645 * se, 2)}

    def total(group, m):
        t = sums[(group, m)]
        est = float(t[0])
        se = (4.0 / 80.0 * float(((t[1:] - est) ** 2).sum())) ** 0.5
        return {'estimate': round(est), 'moe90': round(1.645 * se), 'sample_records': int(unweighted[(group, m)])}

    out = {'year': int(year), 'source_file': os.path.basename(zpath), 'universe': 'ages 18-64, civilian, noninstitutional'}
    for group in ('vision', 'no_vision'):
        lf = sums[(group, 'employed')] + sums[(group, 'unemployed')]
        sums[(group, 'labor_force')] = lf
        unweighted[(group, 'labor_force')] = unweighted[(group, 'employed')] + unweighted[(group, 'unemployed')]
        out[group] = {
            'population': total(group, 'population'),
            'employed': total(group, 'employed'),
            'at_work': total(group, 'at_work'),
            'worked_from_home': total(group, 'worked_from_home'),
            'employment_rate_pct': ratio(group, 'employed', 'population'),
            'unemployment_rate_pct': ratio(group, 'unemployed', 'labor_force'),
            'not_in_labor_force_pct': ratio(group, 'not_in_labor_force', 'population'),
            'work_from_home_share_of_workers_pct': ratio(group, 'worked_from_home', 'at_work'),
            'work_from_home_share_of_population_pct': ratio(group, 'worked_from_home', 'population'),
        }
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'results'), exist_ok=True)
    tag = os.path.basename(zpath).replace('.zip', '')
    dest = os.path.join(here, 'results', f'acs_{year}_{tag}.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    v = out['vision']
    print(f"{year}: vision difficulty employed {v['employment_rate_pct']['estimate']}%, "
          f"work from home {v['work_from_home_share_of_workers_pct']['estimate']}% of workers "
          f"(others {out['no_vision']['work_from_home_share_of_workers_pct']['estimate']}%). Saved {dest}")


if __name__ == '__main__':
    main()
