"""Reproduce sample counts and rates published by the National Research and
Training Center on Blindness and Low Vision from the same 2019 Census file,
as a further check that this study reads the file the way researchers do.

    python check_published.py data/pus_2019.zip

Published figures being reproduced:
  McDonnall, Cmar and McKnight (2022), Social Security disability benefit
  receipt and work: civilian, noninstitutional, ages 18 to 65, with visual
  impairment: 39,934 records representing 4,093,503 people; 19.8 percent
  received Social Security income; 12.4 percent of recipients and 58.8 percent
  of non-recipients worked in the past 12 months.
  McDonnall and Cmar (2024), Underemployment among college graduates: ages 21
  to 64, associate degree or higher, worked in the past 12 months: 695,899
  records representing 69,925,153 people, of whom 6,038 records representing
  610,618 people had a visual impairment.
  McDonnall, Cmar and McKnight (2022), Continuity of employment, and the same
  authors' Earnings paper: civilian, ages 18 to 65, worked in the past 12
  months: 1,521,934 records representing 160,452,588 people, of whom 19,460
  records representing 2,031,140 people had a visual impairment.

The papers do not list every rule used to draw their samples. The two rules
below that are not stated in the papers were found by trial: the benefits
sample matches when active-duty military are removed and no one is removed for
living in group quarters, and the underemployment sample matches when unpaid
family workers are removed. With those rules the record counts match exactly.
"""
import io
import json
import os
import sys
import zipfile

import pandas as pd

COLS = ['AGEP', 'DEYE', 'ESR', 'SCHL', 'WKL', 'SSP', 'COW', 'PWGTP']
t = {k: 0.0 for k in ('ssdi_records', 'ssdi_pop', 'ssdi_recipients', 'rec_worked', 'nonrec', 'nonrec_worked',
                      'ue_records', 'ue_pop', 'ue_vi_records', 'ue_vi_pop', 'wk_records', 'wk_pop', 'wk_vi_records', 'wk_vi_pop')}
with zipfile.ZipFile(sys.argv[1]) as z:
    for name in [n for n in z.namelist() if n.lower().endswith('.csv')]:
        with z.open(name) as f:
            for c in pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), usecols=COLS, chunksize=400_000, low_memory=False):
                a = c[(c['AGEP'] >= 18) & (c['AGEP'] <= 65) & ~c['ESR'].isin([4, 5]) & (c['DEYE'] == 1)]
                rec = pd.to_numeric(a['SSP'], errors='coerce').fillna(0) > 0
                worked = a['WKL'] == 1
                t['ssdi_records'] += len(a)
                t['ssdi_pop'] += a['PWGTP'].sum()
                t['ssdi_recipients'] += a.loc[rec, 'PWGTP'].sum()
                t['rec_worked'] += a.loc[rec & worked, 'PWGTP'].sum()
                t['nonrec'] += a.loc[~rec, 'PWGTP'].sum()
                t['nonrec_worked'] += a.loc[~rec & worked, 'PWGTP'].sum()
                u = c[(c['AGEP'] >= 21) & (c['AGEP'] <= 64) & (c['SCHL'] >= 20) & (c['WKL'] == 1) & (c['COW'] != 8)]
                w = c[(c['AGEP'] >= 18) & (c['AGEP'] <= 65) & ~c['ESR'].isin([4, 5]) & (c['WKL'] == 1)]
                t['wk_records'] += len(w)
                t['wk_pop'] += w['PWGTP'].sum()
                t['wk_vi_records'] += (w['DEYE'] == 1).sum()
                t['wk_vi_pop'] += w.loc[w['DEYE'] == 1, 'PWGTP'].sum()
                t['ue_records'] += len(u)
                t['ue_pop'] += u['PWGTP'].sum()
                t['ue_vi_records'] += (u['DEYE'] == 1).sum()
                t['ue_vi_pop'] += u.loc[u['DEYE'] == 1, 'PWGTP'].sum()
out = {
    'benefits_paper': {
        'records': {'published': 39934, 'mine': int(t['ssdi_records'])},
        'people': {'published': 4093503, 'mine': int(t['ssdi_pop'])},
        'percent_receiving_social_security': {'published': 19.8, 'mine': round(float(100 * t['ssdi_recipients'] / t['ssdi_pop']), 1)},
        'percent_of_recipients_who_worked': {'published': 12.4, 'mine': round(float(100 * t['rec_worked'] / t['ssdi_recipients']), 1)},
        'percent_of_non_recipients_who_worked': {'published': 58.8, 'mine': round(float(100 * t['nonrec_worked'] / t['nonrec']), 1)}},
    'underemployment_paper': {
        'records': {'published': 695899, 'mine': int(t['ue_records'])},
        'people': {'published': 69925153, 'mine': int(t['ue_pop'])},
        'records_with_visual_impairment': {'published': 6038, 'mine': int(t['ue_vi_records'])},
        'people_with_visual_impairment': {'published': 610618, 'mine': int(t['ue_vi_pop'])}},
    'continuity_and_earnings_papers': {
        'records': {'published': 1521934, 'mine': int(t['wk_records'])},
        'people': {'published': 160452588, 'mine': int(t['wk_pop'])},
        'records_with_visual_impairment': {'published': 19460, 'mine': int(t['wk_vi_records'])},
        'people_with_visual_impairment': {'published': 2031140, 'mine': int(t['wk_vi_pop'])}}}
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, 'results', 'published_check.json'), 'w', encoding='utf-8'), indent=1)
for paper, d in out.items():
    for k, v in d.items():
        print(paper, k, v)
