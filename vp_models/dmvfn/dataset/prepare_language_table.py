import os
import json

from collections.abc import Sequence
from absl import app
import tensorflow_datasets as tfds

def main():
    # read entire datasets
    dataset_paths = {
        'language_table_sim': './data/language_table_sim/0.0.1/',
    }

    builder = tfds.builder_from_directory(dataset_paths['language_table_sim'])
    episode_ds = builder.as_dataset(split='train')
    
    



if __name__ == '__main__':

    main()