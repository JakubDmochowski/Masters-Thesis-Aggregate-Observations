from data.tabular.criteo import prepare_observations, get_meta, set_meta
from data.ctr_normalize import CTRNormalize

STD_DEV = 17
CUTOFF = 4 * STD_DEV

prepare_observations(normalize_ctr=CTRNormalize.cutoff,
                     min_count=CUTOFF, remove_outliers=False, with_pairs=True, force=True)
meta = get_meta()
meta['normalizeCTR'] = 'cutoff'
set_meta(meta)
