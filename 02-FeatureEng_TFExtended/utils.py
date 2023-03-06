## UTIL FUNCTIONS TO USE IN TFEXTENDED FOLDER
## -------------------------------------------------------------------##

import tensorflow as tf
from google.protobuf.json_format import MessageToDict

## -------------------------------------------------------------------##
##1. Function to create new folder
def get_records(dataset, num_records):
    '''
    Extracts records from the given dataset.
    :params:
        dataset (TFRecordDataset) - dataset saved by ExampleGen
        num_records (int) - number of records to preview
    '''
    records = []

    for tfrecord in dataset.take(num_records):
        serialized_example = tfrecord.numpy()
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        example_dict = MessageToDict(example)
        records.append(example_dict)
        
    return records

