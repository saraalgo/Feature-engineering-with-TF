# Feature engineering with TF

Functional version 0.1.

## Description

This is a Git Repository designed to explored some TF functionalities to explore data descriptors with [Tensorflow Data Validation (TFDV)](https://github.com/saraalgo/Feature-engineering-with-TF/blob/main/01-DataValidation_TF/DataValidation_TF.ipynb) and [TensorFlow Extended (TFX)](https://github.com/saraalgo/Feature-engineering-with-TF/tree/main/02-FeatureEng_TFExtended). Besides, it provides the code to benchmark the performance of a regression model by applying several [Feature Selection methods](https://github.com/saraalgo/Feature-engineering-with-TF/tree/main/03-FeatureSelection).

## Prerequisites

1. Install python 3.7.9

## Installation

1. Clone the repository in your personal device using the following command:

```sh
git clone https://github.com/saraalgo/Feature-engineering-with-TF.git
```

2. Create and activate python environment if you do not count with the beforehand mentioned Python version. Otherwise, you could skip this step.

```sh
python3.7.9 -m venv Feature-engineering-with-TF/
source bin/activate
```

3. Upgrade pip and install project requirements

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## About the used dataset

It was selected the Kaggle database called **Alcohol Effects On Study**. These are data extracted from secondary school students from two Portuguese schools. The type of data they collect from each student includes grades and demographic, social and academic information, they were collected through questionnaires from the schools. The data used combines two datasets of student performance in two different subjects: mathematics and Portuguese language. The objective of this problem is to be able to predict what your performance will be in the final exam of the course. This problem was previously explored in his original regression version by Cortez 2008.

You can download this dataset from this [link](https://www.kaggle.com/datasets/whenamancodes/alcohol-effects-on-study).

## Funcitonalities

The code is adapted to the before mentioned dataset, however, it could be adapted according with the neccesities. A short description of what can be found on each folder of this repository:

- **01-DataValidation_TF**: Case of use of how to apply the TensorFlow Data Validation ([**TFDV**](https://www.tensorflow.org/tfx/data_validation/get_started)) functions to a dataset.
- **02-FeatureEng_TFExtended**: Guide of how to apply the same methods of TFDV, but using TensorFlow Extended ([TFX](https://www.tensorflow.org/tfx)). Besides, the main functions to apply the component Transform of this packages are designed in the corresponding utils file.
- **03-FeatureSelection**: Benchmarking of approximations to apply Feature Selection for this regression problem, exploring filter, wrapper and embbeded methods. Using for these tasks the [sklearn.feature_selection package](https://scikit-learn.org/stable/modules/feature_selection.html).