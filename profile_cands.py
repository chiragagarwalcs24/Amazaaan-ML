import time
import pandas as pd
import joblib

from src.blocking import _build_index, _build_country_index, _candidates_for_one

print('Loading dataset...')
sources = joblib.load('output/test_normalized.pkl')
s1, s2, s3 = sources['s1'], sources['s2'], sources['s3']

print('Building index...')
idx_s2 = _build_index(s2)
idx_s3 = _build_index(s3)
ctry_s2 = _build_country_index(s2)
ctry_s3 = _build_country_index(s3)

valid_all = set(s2['entity_id']) | set(s3['entity_id'])
t1_records = s1.to_dict(orient='records')
chunk = t1_records[:1000]

print('Running 1000 items...')
t0 = time.time()
cand_rows = []
for s1_row in chunk:
    s1_id   = s1_row['entity_id']
    name_n  = s1_row.get('name_norm', '')
    addr_n  = s1_row.get('addr_norm', '')
    country = s1_row.get('country', '')
    cands = _candidates_for_one(
        name_n, addr_n, country,
        idx_s2, idx_s3, ctry_s2, ctry_s3, 100,
    )
    cands = [c for c in cands if c in valid_all]
    for c_id in cands:
        cand_rows.append({'source1_entity_id': s1_id, 'candidate_entity_id': c_id})

t1 = time.time()
print(f'Candidate generation for 1000 items took: {t1-t0:.2f}s (Extrapolates to {(t1-t0)*10:.2f}s for 10k chunk)')
print(f'Generated {len(cand_rows)} pairs.')
