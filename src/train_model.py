# pandas to load the dataset
import pandas as pd
# for the machine learning models
# we have to import these modules explicitly as sklearn does not import its own submodules
import sklearn.model_selection as ms
import sklearn.neural_network as nn
import sklearn.svm as svm
import sklearn.metrics as metrics
# to serialise the model when it's trained
import pickle as pkl

import matplotlib

# location of the serialised model to save to
model_location = '../models/mlp.pkl'

# load training dataset
dataset_location = '../data/wdbc.csv'
original_dataset = pd.read_csv(dataset_location)

# split dataset into features and class labels
features = original_dataset.values[:, 1:31]
labels = original_dataset.values[:, 31]

# split dataset into training and validation sets
# test_size = percentage used for validation
# random state = random seed for shuffling
features_train, features_test, labels_train, labels_test = ms.train_test_split(features,
                                                                               labels,
                                                                               test_size=0.3,
                                                                               random_state=123)

# establish the model
# untrained_model = nn.MLPClassifier(activation='tanh', solver='lbfgs', max_iter=400, momentum=0.9)
# untrained_model = svm.LinearSVC()
untrained_model = svm.SVC(kernel='rbf', gamma='scale')

# train the model
trained_model = untrained_model.fit(features_train, labels_train)

# validate that our model is accurate enough
model_accuracy = trained_model.score(features_test, labels_test)
predicted_labels = trained_model.predict(features_test)
model_confusion = metrics.confusion_matrix(labels_test, predicted_labels)

# just for us
print(model_accuracy)
print(model_confusion)

# serialise the trained model so we can use it in our service
with open(model_location, 'wb') as handle:
    pkl.dump(trained_model, handle)
