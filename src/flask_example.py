# the actual web service
import flask as fl
# to deserialise the pre-trained model
import pickle as pkl

app = fl.Flask(__name__)

# location of the serialised model to load
model_location = 'models/model_name'

# load serialised model
with open(model_location, 'rb') as handle:
    model = pkl.load(handle)


# take a POST request at the specified endpoint with simple JSON payload and
# make a prediction of cancer based on the values provided
# note that we don't need sklearn anymore, it's all contained in the pickle
@app.route('predict-cancer', methods=['POST'])
def predict_cancer():
    data = fl.request.get_json()
    prediction = model.predict()
    result = prediction[0]
    return fl.jsonify(result)


# run the Flask app
if __name__ == "__main__":
    app.run(port=9000, debug=True)