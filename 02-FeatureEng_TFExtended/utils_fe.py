## UTIL FUNCTIONS TO USE IN TFEXTENDED FOLDER
## -------------------------------------------------------------------##

import tensorflow as tf
from google.protobuf.json_format import MessageToDict
import tensorflow_transform as tft

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

## -------------------------------------------------------------------##
## MODIFY IN CASE THE DATASET CHANGES
##2. Gather the features according with the desired transformation in FE

# Features to apply the z-score
ZSCORE_FEATURE_KEYS = ['Medu', 'Fedu', 'traveltime',
                      'studytime', 'failures', 'famrel',
                      'freetime', 'goout', 'Dalc',
                      'Walc', 'health', 'G1', 'G2']
# Features to bucketize
BUCKET_FEATURE_KEYS = ['sex', 'address', 'famsize',
                       'Pstatus', 'guardian', 'schoolsup',
                       'famsup', 'paid', 'activities',
                       'nursery', 'higher', 'internet',
                       'romantic']
# Number of buckets for each feature.
FEATURE_BUCKET_COUNT = {'sex': 2,
                        'address': 2,
                        'famsize': 2,
                        'Pstatus': 2,
                        'guardian': 3,
                        'schoolsup': 2,
                        'famsup': 2,
                        'paid': 2,
                        'activities': 2,
                        'nursery': 2,
                        'higher': 2,
                        'internet': 2,
                        'romantic': 2}
# Scale from 0 to 1
SCALE_FEATURE_KEYS = ['absences']
# Number of vocabulary terms used for encoding VOCAB_FEATURE_KEYS
VOCAB_SIZE = 20
# Features with string data types that will be converted to indices
VOCAB_FEATURE_KEYS = ['Mjob', 'Fjob', 'reason']
# Int data type that will be kept equal
KEEP_FEATURE_KEYS = ['age']
# Output feature
OUTPUT_KEY = 'G3'

def transformed_name(key):
    return key + '_t'

## 3. Function to Transform each feature with TFT

def feature_eng(features):
    """Function for preprocessing inputs with tf.transform
    :params:
        features: Map from feature keys to raw not-yet-transformed features.
    :return:
        outputs: Map from string feature key to transformed feature operations.
    """
    outputs = {}

    ### START CODE HERE
    
     # Scale these features to the z-score.
    for key in ZSCORE_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(features[key])

    # Scale these feature/s from 0 to 1
    for key in SCALE_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_0_1(features[key])
            
    # Transform the strings into indices 
    # hint: use the VOCAB_SIZE and OOV_SIZE to define the top_k and num_oov parameters
    for key in VOCAB_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
                                            features[key], top_k=VOCAB_SIZE)

    # Bucketize the feature
    for key in BUCKET_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.bucketize(
                                            features[key], 
                                            FEATURE_BUCKET_COUNT[key])

    # No tft function needed.
    for key in KEEP_FEATURE_KEYS:
        outputs[transformed_name(key)] = features[key]

    # Use `tf.cast` to cast the label key to float32 and fill in the missing values.
    outputs[transformed_name(OUTPUT_KEY)] = tf.cast(features[OUTPUT_KEY], tf.float32)                                                       

    ### END CODE HERE
    return outputs