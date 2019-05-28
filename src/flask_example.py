# the actual web service
import flask as fl
# to deserialise the pre-trained model
import pickle as pkl
# to format the input data in a way that the model understands
import numpy as np

app = fl.Flask(__name__)

# location of the serialised model to load
model_location = '../models/mlp.pkl'

# load serialised model
with open(model_location, 'rb') as handle:
    model = pkl.load(handle)


# take a POST request at the specified endpoint with simple JSON payload and
# make a prediction of cancer based on the values provided
# note that we don't need sklearn anymore, it's all contained in the pickle
@app.route('/predict-cancer', methods=['POST'])
def predict_cancer():
    # read in the JSON request
    data = fl.request.get_json()
    # convert it to a 2d array so that predict() understands it
    features = [np.array(data['features'])]
    # do the actual prediction
    prediction = model.predict(features)
    # get the result
    result = prediction[0]
    return fl.jsonify(result)


# run the Flask app
if __name__ == "__main__":
    app.run(port=9000, debug=True)
