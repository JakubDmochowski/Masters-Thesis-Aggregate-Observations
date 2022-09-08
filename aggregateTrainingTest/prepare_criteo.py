from data.tabular.criteo import prepareObservations, CTRNormalize, getMeta, setMeta

STD_DEV = 17
CUTOFF = 4 * STD_DEV


def validObservation(entry) -> bool:
    feature_value, feature_id, count, clicks, sales = entry
    return float(count) >= CUTOFF


prepareObservations(normalizeCTR=CTRNormalize.cutoff,
                    filterObservations=validObservation, removeOutliers=False, force=True)
meta = getMeta()
meta['filter'] = f"count below {CUTOFF}"
meta['normalizeCTR'] = 'cutoff'
setMeta(meta)
