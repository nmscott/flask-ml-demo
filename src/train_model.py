# pandas to load the dataset
import pandas as pd
# for the machine learning models
import sklearn as sk
# to serialise the model when it's trained
import pickle as pkl

# location of the serialised model to save to
model_location = 'models/model_name'

# load training dataset
dataset_location = 'data/name.csv'
original_dataset = pd.read_csv(dataset_location)

# split dataset into features and class labels
features = original_dataset.values[:, 0:3]
labels = original_dataset.values[:, 4]

# split dataset into training and validation sets
# test_size = percentage used for validation
# random state = random seed for shuffling
features_train, features_test, labels_train, labels_test = sk.model_selection.train_test_split(features, labels,
                                                                                               test_size=0.3,
                                                                                               random_state=123)

# establish the model


# train the model


# validate that our model is accurate enough


# serialise the trained model so we can use it in our service
with open(model_location, 'wb') as handle:
    pkl.dump(handle)
