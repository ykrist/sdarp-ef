#!/usr/bin/env python
from utils.data import indices
from sdarp import NAMED_DATASET_INDICES
import fraglib

if __name__ == '__main__':
    indices.main_func('sdarp', NAMED_DATASET_INDICES, index_to_name=fraglib.sdarp.instance.index_to_name)
