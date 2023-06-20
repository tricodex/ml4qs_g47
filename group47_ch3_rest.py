##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

"""
Command Line Arguments: The script uses the argparse library to handle command line arguments. The --mode argument is used to determine which preprocessing task to perform. The options are 'imputation', 'lowpass', 'PCA', 'kalman', and 'final'.

Data Loading: The script loads a dataset from a specified location. The dataset is expected to be a CSV file with a datetime index.

Mode Selection: Depending on the --mode argument, the script performs one of the following tasks:

Imputation: The script imputes missing values in the 'acc_phone_x' column of the dataset using three different methods: mean, median, and interpolation. The results are then plotted for comparison.

Kalman Filtering: The script applies a Kalman filter to the 'acc_phone_x' attribute of the original dataset and plots the results.

Low-Pass Filtering: The script applies a low-pass filter to the 'acc_phone_x' column of the dataset to reduce the importance of data above 1.5 Hz. The filtered data is then plotted.

PCA: The script first imputes missing values using interpolation. Then, it applies PCA to the dataset and plots the explained variance of the principal components. It then applies PCA with 7 components (as determined to be optimal) to the dataset and plots the results.

Final: In the final mode, the script first imputes missing values using interpolation. Then, it applies a low-pass filter to certain columns of the dataset. After that, it applies PCA with 7 components to the dataset. The final preprocessed dataset is then plotted and saved to a CSV file. (THIS IS AGAIN THE DEFAULT MODE)

Visualization: The VisualizeDataset class is used throughout the script to visualize the results of each preprocessing task.

Saving the Results: The final preprocessed dataset is saved to a CSV file.

"""

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter3.DataTransformation import LowPassFilter
from Chapter3.DataTransformation import PrincipalComponentAnalysis
from Chapter3.ImputationMissingValues import ImputationMissingValues
from Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./datasets/group47/dataset/intermediate_datafiles/')     
DATASET_FNAME = 'chapter3_group47_result_outliers.csv'
RESULT_FNAME = 'chapter3_group47_result_final.csv'
ORIG_DATASET_FNAME = 'chapter2_group47_result.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():

    print_flags()

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (
        dataset.index[1] - dataset.index[0]).microseconds/1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    if FLAGS.mode == 'imputation':
        # Let us impute the missing values and plot an example.
       
        imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'acc_phone_x')       
        imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'acc_phone_x')
        imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'acc_phone_x')
        
        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'acc_phone_x',
                                    imputed_mean_dataset['acc_phone_x'], 
                                    imputed_median_dataset['acc_phone_x'],
                                    imputed_interpolation_dataset['acc_phone_x'])

    elif FLAGS.mode == 'kalman':
        # Using the result from Chapter 2, let us try the Kalman filter on the acc_phone_x attribute and study the result.
        try:
            original_dataset = pd.read_csv(
            DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
            original_dataset.index = pd.to_datetime(original_dataset.index)
        except IOError as e:
            print('File not found, try to run previous crowdsignals scripts first!')
            raise e

        KalFilter = KalmanFilters()
        kalman_dataset = KalFilter.apply_kalman_filter(
            original_dataset, 'acc_phone_x')
        DataViz.plot_imputed_values(kalman_dataset, [
                                    'original', 'kalman'], 'acc_phone_x', kalman_dataset['acc_phone_x_kalman'])
        DataViz.plot_dataset(kalman_dataset, ['acc_phone_x', 'acc_phone_x_kalman'], ['exact', 'exact'], ['line', 'line'])

        # We ignore the Kalman filter output for now...

    elif FLAGS.mode == 'lowpass':
        
        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000)/milliseconds_per_instance
        cutoff = 0.427

        # Let us study acc_phone_x:
        new_dataset = LowPass.low_pass_filter(copy.deepcopy(
            dataset), 'acc_phone_x', fs, cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4*len(new_dataset.index)):int(0.43*len(new_dataset.index)), :],
                             ['acc_phone_x', 'acc_phone_x_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'PCA':

        #first impute again, as PCA can not deal with missing values       
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
        pc_values = PCA.determine_pc_explained_variance(dataset, selected_predictor_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_predictor_cols)+1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'])

        # We select 7 as the best number of PC's as this explains most of the variance
        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(dataset, ['pca_', 'label'], ['like', 'like'], ['line', 'points'])

    elif FLAGS.mode == 'final':
        # Now, for the final version. 
        # We first start with imputation by interpolation
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
        periodic_measurements = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z']

        # Let us apply a lowpass filter and reduce the importance of the data above 0.42 Hz
        # Determine the sampling frequency.
        # Let us apply a lowpass filter and reduce the importance of the data above the optimal cutoff frequency
        # Determine the sampling frequency.
        fs = float(1000)/milliseconds_per_instance

        # Dictionary of optimal cutoff frequencies for each type of measurement
        optimal_cutoffs = {
            'acc_phone_x': 0.42703990601536507,
            'acc_phone_y': 0.42703990601536507,
            'acc_phone_z': 0.42703990601536507,
            'lin_acc_phone_x': 0.47291473535472006,
            'lin_acc_phone_y': 0.47291473535472006,
            'lin_acc_phone_z': 0.47291473535472006,
            'mag_phone_x': 0.12012946653705107,
            'mag_phone_y': 0.12012946653705107,
            'mag_phone_z': 0.12012946653705107
        }

        for col in periodic_measurements:
            # Get the optimal cutoff frequency for this measurement
            cutoff = optimal_cutoffs[col]
            # Apply the low-pass filter with the optimal cutoff frequency
            dataset = LowPass.low_pass_filter(dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']


        # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset
        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
        n_pcs = 7
        dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

        # And the overall final dataset:
        # And the overall final dataset:
        print(dataset)  # Check if the dataset is empty
        print(dataset.columns)  # Check if the columns exist
        print(dataset[['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7']].dropna().empty)  # Check if the columns are all NaNs
        # Check if the columns are all NaNs

        DataViz.plot_dataset(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
                     ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'])



        # Store the final outcome.
        dataset.to_csv(DATA_PATH / RESULT_FNAME)



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'imputation', 'PCA', 'final'])

   
    FLAGS, unparsed = parser.parse_known_args()

    main()
