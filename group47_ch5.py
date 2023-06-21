##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import util.util as util
from util.VisualizeDataset import VisualizeDataset

import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def main():

    # As usual, we set our program constants, read the input file and initialize a visualization object.
    DATA_PATH = Path('./datasets/group47/dataset/intermediate_datafiles/')  # Adjusted path
    DATASET_FNAME = 'chapter4_group47_result.csv'  # Adjusted file name
    RESULT_FNAME = 'chapter5_group47_result.csv'  # Adjusted result file name

    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    DataViz = VisualizeDataset(__file__)

    clusteringNH = NonHierarchicalClustering()
    clusteringH = HierarchicalClustering()

    # ... Rest of the code remains the same ...

    if FLAGS.mode == 'final':
        # And we select the outcome dataset of the knn clustering....
        clusteringNH = NonHierarchicalClustering()

        dataset = clusteringNH.k_means_over_instances(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z'], FLAGS.k, 'default', 50, 50)
        DataViz.plot_clusters_3d(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z'], 'cluster', ['label'])
        DataViz.plot_silhouette(dataset, 'cluster', 'silhouette')
        util.print_latex_statistics_clusters(
            dataset, 'cluster', ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z'], 'label')
        del dataset['silhouette']

        dataset.to_csv(DATA_PATH / RESULT_FNAME)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, kmeans, kmediods, hierarchical or aggloromative. \
                        'kmeans' to study the effect of kmeans on a selection of variables \
                        'kmediods' to study the effect of kmediods on a selection of variables \
                        'agglomerative' to study the effect of agglomerative clustering on a selection of variables  \
                        'final' kmeans with an optimal level of k is used for the next chapter", choices=['kmeans', 'kmediods', 'agglomerative', 'final'])

    parser.add_argument('--k', type=int, default=6,
                        help="The selected k number of means used in 'final' mode of this chapter' \
                        ")

    FLAGS, unparsed = parser.parse_known_args()

    main()