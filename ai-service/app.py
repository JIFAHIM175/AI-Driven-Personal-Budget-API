from flask import Flask, request, jsonify
from lstm_model import predict_next_value

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        category = data.get("category", None)
        prediction = predict_next_value(category)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)


