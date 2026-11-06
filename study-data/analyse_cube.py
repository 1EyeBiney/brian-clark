"""Read the yearly cubes and print every follow-up table with margins of error.

    python analyse_cube.py            prints tables and writes results/followup.json

Margins of error are 90 percent, from the 80 replicate weights. For figures
pooled over several years, each year's replicate estimates are combined as
independent samples.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
REPS = [f'PWGTP{i}' for i in range(1, 81)]
W = ['PWGTP'] + REPS

cube = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(here, 'results', 'cube_*_pus_*.csv')))], ignore_index=True)
years = sorted(cube['year'].unique())
out = {'years': [int(y) for y in years]}


def tot(df):
    return df[W].sum().to_numpy(dtype='float64')


def ratio(num_df, den_df, by_year_pool=False):
    """Percent with 90% MOE. If the frames span several years, variance is summed over years."""
    ys = sorted(set(den_df['year'].unique()))
    N = sum(tot(num_df[num_df['year'] == y])[0] for y in ys)
    D = sum(tot(den_df[den_df['year'] == y])[0] for y in ys)
    if D == 0:
        return None
    est = 100.0 * N / D
    var = 0.0
    for y in ys:
        n, d = tot(num_df[num_df['year'] == y]), tot(den_df[den_df['year'] == y])
        # replicate estimates of the pooled ratio, varying this year only
        r = 100.0 * (N - n[0] + n[1:]) / (D - d[0] + d[1:])
        var += 4.0 / 80.0 * float(((r - est) ** 2).sum())
    return {'pct': round(est, 1), 'moe': round(1.645 * var ** 0.5, 1), 'records': int(den_df['records'].sum())}


def show(label, r):
    if r is None:
        print(f'  {label}: no data')
    else:
        print(f"  {label}: {r['pct']}% (+/- {r['moe']}, {r['records']:,} records)")


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k].isin(v)] if isinstance(v, (list, tuple)) else df[df[k] == v]
    return df


atwork = ['home', 'onsite']
res = {}

print('=== A. Work from home share of people at work, by education')
for y in years:
    for vis in (1, 0):
        for col in (1, 0):
            d = sel(cube, year=y, vision=vis, college=col, place=atwork)
            r = ratio(sel(d, place='home'), d)
            key = f'wfh_{y}_vision{vis}_college{col}'
            res[key] = r
            show(key, r)

print('=== B. Employment rate by education')
for y in years:
    for vis in (1, 0):
        for col in (1, 0):
            d = sel(cube, year=y, vision=vis, college=col)
            r = ratio(sel(d, status='employed'), d)
            key = f'emp_{y}_vision{vis}_college{col}'
            res[key] = r
            show(key, r)

print('=== C. Share with a bachelor\'s degree (whole group, and those employed)')
for y in years:
    for vis in (1, 0):
        d = sel(cube, year=y, vision=vis)
        key = f'college_share_{y}_vision{vis}'
        res[key] = ratio(sel(d, college=1), d)
        show(key, res[key])
        e = sel(d, status='employed')
        key = f'college_share_employed_{y}_vision{vis}'
        res[key] = ratio(sel(e, college=1), e)
        show(key, res[key])

OCC = ['management_professional', 'office_admin', 'sales', 'service', 'construction_production_transport']
print('=== D. Occupation: distribution of people at work, and work-from-home share within each')
for y in (years[0], years[-1]):
    for vis in (1, 0):
        d = sel(cube, year=y, vision=vis, place=atwork)
        for o in OCC:
            k1 = f'occ_dist_{y}_vision{vis}_{o}'
            res[k1] = ratio(sel(d, occ=o), d)
            k2 = f'occ_wfh_{y}_vision{vis}_{o}'
            res[k2] = ratio(sel(d, occ=o, place='home'), sel(d, occ=o))
            print(f"  {y} vision{vis} {o}: {res[k1]['pct']}% of workers; of them {res[k2]['pct']}% (+/- {res[k2]['moe']}) work from home")

print('=== E. What if workers with vision difficulty had the same occupation mix as others? (last year)')
y = years[-1]
d1 = sel(cube, year=y, vision=1, place=atwork)
d0 = sel(cube, year=y, vision=0, place=atwork)
std = 0.0
for o in OCC + ['none']:
    share0 = tot(sel(d0, occ=o))[0] / tot(d0)[0]
    den = tot(sel(d1, occ=o))[0]
    rate1 = tot(sel(d1, occ=o, place='home'))[0] / den if den else 0
    std += share0 * rate1
res['wfh_vision_with_others_occupation_mix'] = round(100 * std, 1)
print(f"  actual {res[f'wfh_{y}_vision1_college1']['pct'] if False else round(100 * tot(sel(d1, place='home'))[0] / tot(d1)[0], 1)}%, with others' occupation mix {round(100 * std, 1)}%, others actual {round(100 * tot(sel(d0, place='home'))[0] / tot(d0)[0], 1)}%")

print('=== F. Indiana')
last3 = [yy for yy in years if yy >= 2022]
for label, ys in (('first_year', [years[0]]), ('pooled_2022_on', last3)):
    for vis in (1, 0):
        d = sel(cube, indiana=1, vision=vis, year=ys)
        res[f'indiana_emp_{label}_vision{vis}'] = ratio(sel(d, status='employed'), d)
        show(f'indiana employment {label} vision{vis}', res[f'indiana_emp_{label}_vision{vis}'])
        w = sel(d, place=atwork)
        res[f'indiana_wfh_{label}_vision{vis}'] = ratio(sel(w, place='home'), w)
        show(f'indiana work from home {label} vision{vis}', res[f'indiana_wfh_{label}_vision{vis}'])
dd = sel(cube, indiana=1, vision=1, year=[years[-1]])
res['indiana_vision_population_last'] = int(tot(dd)[0])
print('  Indiana working-age people with vision difficulty, last year:', res['indiana_vision_population_last'])

print('=== G. Who is in the group: age and other disabilities')
for y in (years[0], years[-1]):
    d = sel(cube, year=y, vision=1)
    for a in ('18-34', '35-49', '50-64'):
        res[f'age_share_{y}_{a}'] = ratio(sel(d, age=a), d)
        show(f'{y} age {a} share of vision group', res[f'age_share_{y}_{a}'])
    res[f'vision_only_share_{y}'] = ratio(sel(d, other_dis=0), d)
    show(f'{y} vision difficulty only (no other difficulty)', res[f'vision_only_share_{y}'])
    for od in (0, 1):
        g = sel(d, other_dis=od)
        res[f'emp_{y}_vision_otherdis{od}'] = ratio(sel(g, status='employed'), g)
        show(f'{y} employment, other difficulty={od}', res[f'emp_{y}_vision_otherdis{od}'])
        res[f'pop_{y}_vision_otherdis{od}'] = int(tot(g)[0])
        print(f'    population {int(tot(g)[0]):,}')
    for a in ('18-34', '35-49', '50-64'):
        g = sel(d, age=a)
        res[f'emp_{y}_vision_age_{a}'] = ratio(sel(g, status='employed'), g)
        show(f'{y} employment age {a}', res[f'emp_{y}_vision_age_{a}'])

print('=== H. Self-employment and full-time among people at work, by place')
for y in (years[0], years[-1]):
    for vis in (1, 0):
        for pl in atwork:
            d = sel(cube, year=y, vision=vis, place=pl)
            res[f'selfemp_{y}_vision{vis}_{pl}'] = ratio(sel(d, selfemp=1), d)
            res[f'fulltime_{y}_vision{vis}_{pl}'] = ratio(sel(d, fulltime=1), d)
            print(f"  {y} vision{vis} {pl}: self-employed {res[f'selfemp_{y}_vision{vis}_{pl}']['pct']}%, full-time {res[f'fulltime_{y}_vision{vis}_{pl}']['pct']}%")

print('=== I. Median earnings (people at work)')
for f in sorted(glob.glob(os.path.join(here, 'results', 'earn_*_pus_*.json'))):
    e = json.load(open(f))
    res[f"earn_{e['year']}"] = e
    print(' ', e['year'], {k: (v['median_earnings'], v['median_hours']) for k, v in e.items() if isinstance(v, dict)})

print('=== J. People who already hold a bachelor\'s degree (last year)')
y = years[-1]
for a in ('18-34', '35-49', '50-64'):
    for vis in (1, 0):
        d = sel(cube, year=y, vision=vis, college=1, age=a)
        res[f'emp_{y}_vision{vis}_college1_age_{a}'] = ratio(sel(d, status='employed'), d)
        show(f'{y} degree holders, vision{vis}, age {a}, employed', res[f'emp_{y}_vision{vis}_college1_age_{a}'])
d = sel(cube, year=y, vision=1, college=1)
res[f'pop_{y}_vision1_college1'] = int(tot(d)[0])
res[f'not_employed_{y}_vision1_college1'] = int(tot(sel(d, status=['unemployed', 'nilf']))[0])
res[f'nilf_{y}_vision1_college1'] = int(tot(sel(d, status='nilf'))[0])
print('  degree holders with vision difficulty:', res[f'pop_{y}_vision1_college1'], 'not employed:',
      res[f'not_employed_{y}_vision1_college1'], 'of them not looking:', res[f'nilf_{y}_vision1_college1'])
for od in (0, 1):
    g = sel(d, other_dis=od)
    res[f'emp_{y}_vision1_college1_otherdis{od}'] = ratio(sel(g, status='employed'), g)
    show(f'{y} degree holders with vision difficulty, other difficulty={od}, employed', res[f'emp_{y}_vision1_college1_otherdis{od}'])

out['results'] = res
with open(os.path.join(here, 'results', 'followup.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=1)
