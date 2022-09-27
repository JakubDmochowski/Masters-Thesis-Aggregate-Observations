from data.tabular.criteo import prepareObservations, CTRNormalize, getMeta, setMeta

STD_DEV = 17
CUTOFF = 4 * STD_DEV


prepareObservations(normalizeCTR=CTRNormalize.cutoff,
                    minCount = CUTOFF, removeOutliers=False, withPairs=True, force=True)
meta = getMeta()
meta['normalizeCTR'] = 'cutoff'
setMeta(meta)
