# pandas to load the dataset
import pandas as pd
# for the machine learning models
# we have to import these modules explicitly as sklearn does not import its own submodules
import sklearn.model_selection as ms
import sklearn.neural_network as nn
# to serialise the model when it's trained
import pickle as pkl

# location of the serialised model to save to
model_location = '../models/mlp.pkl'

# load training dataset
dataset_location = '../data/wdbc.csv'
original_dataset = pd.read_csv(dataset_location)

# split dataset into features and class labels
features = original_dataset.values[:, 1:31]
labels = original_dataset.values[:, 31]

print(features)
print(features.shape)
print(labels)

# split dataset into training and validation sets
# test_size = percentage used for validation
# random state = random seed for shuffling
features_train, features_test, labels_train, labels_test = ms.train_test_split(features,
                                                                               labels,
                                                                               test_size=0.3,
                                                                               random_state=123)

# establish the model
mlp = nn.MLPClassifier(activation='tanh', solver='lbfgs', max_iter=200, momentum=0.9)

# train the model
model = mlp.fit(features_train, labels_train)

# validate that our model is accurate enough
model_accuracy = mlp.score(features_test, labels_test)

print(model_accuracy)

# serialise the trained model so we can use it in our service
with open(model_location, 'wb') as handle:
    pkl.dump(model, handle)
