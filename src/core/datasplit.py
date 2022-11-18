import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratifiedsplit(dataset, dataset_labels, label_string: str, bin_size: int, second_split_size = 0.7) -> (np.array, np.array, np.array, np.array):
    """Split datset into two subsets with customized split size (second_split_size) and stratify based on label distribution.
    :param dataset: pd.DataFrame, dataset to be split
    :param dataset_labels: pd.Series, series of labels for training/testing
    :param label_string: name of dataset column with labels
    :param bin_size: number of values to group into each bin (for a label distribution of 0-100)
    :param second_split_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the second split/subset
    :return: np.array, split1, split1 labels, split2, split2 labels
    """
    #categorize EF values into 10 bins/classes 
    bin_classes = np.arange(0,105,bin_size) #[0,5,10....100] --> len -1 = 20 bins
    dataset['BinClass'] = np.digitize(dataset[label_string], bin_classes) 

    #check that >2 members/bin (required for train_test_split to split data based on distribution of EF bins)
    if min(dataset['BinClass'].value_counts()) < 2:
        raise Exception("Use a larger bin size. The minimum number of members per class/bin cannot be less than 2.")

    #split dataset and stratify based on EF bins/classes
    split_1, split_2, split_1_labels, split_2_labels = train_test_split(dataset, dataset_labels.to_list(), test_size = second_split_size, random_state = 42, stratify = dataset['BinClass'].to_list())

    return split_1, split_2, split_1_labels, split_2_labels