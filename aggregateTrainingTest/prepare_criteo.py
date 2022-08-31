from data.tabular.criteo import prepareObservations
import os

dirpath = os.getcwd() + "\\datasets\\criteo\\prepared"

prepareObservations(force = True, ctr_norm='cutoff')
