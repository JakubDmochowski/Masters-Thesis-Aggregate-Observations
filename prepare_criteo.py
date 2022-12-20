from data.data_generator import DataGenerator
from data.tabular.criteo import download_aggregated_pairs, CriteoDataGraph
from data.ctr_normalize import CTRNormalize
import os

prepared_directory = os.getcwd() + "/datasets/criteo/prepared"
if not os.path.exists(prepared_directory):
    os.makedirs(prepared_directory)

data_gen_dest = os.getcwd() + "/datasets/criteo/prepared/generated.csv"

raw_directory = os.getcwd() + "/datasets/criteo/raw"
pairs_filepath = raw_directory + "/aggregated_noisy_data_pairs.csv"
if not os.path.exists(pairs_filepath):
    download_aggregated_pairs(raw_directory)

data_graph = CriteoDataGraph()
data_graph.prep()
gen = DataGenerator(data_graph=data_graph, no_attributes=19,
                    ctr_normalize=CTRNormalize.cutoff)
gen.generate_data(500000, filename=data_gen_dest)
