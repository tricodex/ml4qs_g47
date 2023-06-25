##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

"""
The script sets up paths and filenames for data storage and retrieval.

It creates necessary directories if they don't exist.

It defines two granularities (time intervals) for data sampling: 1 minute and 250 milliseconds.

For each granularity, it creates a dataset by aggregating accelerometer, linear acceleration, and magnetometer data. The data is averaged over each time interval.

It visualizes the dataset by generating boxplots and line plots.

It prints some statistical information about the dataset.

If more than one granularity was used, it compares the statistics of the datasets and prints them in a LaTeX table format.

It saves the final dataset to a CSV file.

"""

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import pandas as pd

DATASET_PATH = Path('./datasets/group47/dataset/')
RESULT_PATH = Path('./datasets/group47/dataset/intermediate_datafiles/')
RESULT_FNAME = 'chapter2_group47_result.csv'

GRANULARITIES = [7241, 1000, 60000, 250] # We might change this

[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

datasets = []
for milliseconds_per_instance in GRANULARITIES:
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the accelerometer data
    dataset.add_numerical_dataset('accelerometer_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'acc_phone_')

    # Add the linear acceleration data
    dataset.add_numerical_dataset('linear_acceleration_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'lin_acc_phone_')

    # Add the magnetometer data
    dataset.add_numerical_dataset('magnetometer_phone.csv', 'timestamps', ['x','y','z'], 'avg', 'mag_phone_')

    # Add the label data
    dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')



    dataset = dataset.data_table

    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ['acc_phone_x','acc_phone_y','acc_phone_z', 'lin_acc_phone_x','lin_acc_phone_y','lin_acc_phone_z', 'mag_phone_x','mag_phone_y','mag_phone_z'])

    # Plot all data
    DataViz.plot_dataset(dataset, ['acc_', 'lin_acc_', 'mag_', 'label'],
                                  ['like', 'like', 'like', 'like'],
                                  ['line', 'line', 'line', 'points'])

    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    dataset.to_csv(RESULT_PATH / (str(milliseconds_per_instance)+RESULT_FNAME))

util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])



print('The code has run through successfully!')
