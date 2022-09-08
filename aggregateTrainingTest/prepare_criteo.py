from data.tabular.criteo import prepareObservations, CTRNormalize, getMeta, setMeta, saveMeta

STD_DEV = 17
CUTOFF = 4 * STD_DEV


def validObservation(entry) -> bool:
    feature_value, feature_id, count, clicks, sales = entry
    return float(count) >= CUTOFF


filename = f"observations_{str(CUTOFF)}"

prepareObservations(filename=filename, normalizeCTR=CTRNormalize.cutoff,
                    filterObservations=validObservation, removeOutliers=False, force=True)
meta = getMeta()
meta['filter'] = f"count below {CUTOFF}"
meta['normalizeCTR'] = 'cutoff'
setMeta(meta)
saveMeta(filename)
