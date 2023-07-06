##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./datasets/group47/dataset/intermediate_datafiles/')  # Adjusted path
DATASET_FNAME = 'chapter3_group47_result_final.csv'  # Adjusted file name
RESULT_FNAME = 'chapter4_group47_result.csv'  # Adjusted result file name

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print_flags()
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    # milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000
    # milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).total_seconds() * 1000
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).total_seconds()



    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    

    if FLAGS.mode == 'final':
        # ws = int(float(0.5*60000)/milliseconds_per_instance)
        # fs = float(1000)/milliseconds_per_instance
        # ws = int(float(0.5*60)/milliseconds_per_instance)
        ws = int(float(0.5*60)/milliseconds_per_instance)

        fs = float(1)/milliseconds_per_instance


        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')
        # TODO: Add your own aggregation methods here
        
        plot_columns = ['acc_phone_x', 'gyr_phone_x', 'hr_watch_rate', 'light_phone_lux', 'mag_phone_x', 'press_phone_', 'pca_1', 'label']
        plot_columns = [col for col in plot_columns if col in dataset.columns and not dataset[col].empty]  # Ensure columns exist and are not empty

        DataViz.plot_dataset(dataset, plot_columns, ['like']*len(plot_columns), ['line']*len(plot_columns))
     
        CatAbs = CategoricalAbstraction()
        
        dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)


        periodic_predictor_cols = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z']

        #dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)
        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10)/milliseconds_per_instance), fs)

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]
        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'lin_acc_phone_x', 'lin_acc_phone_y', 'lin_acc_phone_z', 'mag_phone_x', 'mag_phone_y', 'mag_phone_z', 'pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7'], ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'])

        print("--- %s seconds ---" % (time.time() - start_time))

        # Store the final outcome.
        dataset.to_csv(DATA_PATH / RESULT_FNAME)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final']) 

    FLAGS, unparsed = parser.parse_known_args()
    
    main()

    