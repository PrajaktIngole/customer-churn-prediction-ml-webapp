from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and columns
model = pickle.load(open("churn_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         data = request.form.to_dict()
#         df = pd.DataFrame([data])

#         # Convert numeric fields
#         df['tenure'] = float(df['tenure'])
#         df['MonthlyCharges'] = float(df['MonthlyCharges'])
#         df['SeniorCitizen'] = int(df['SeniorCitizen'])

#         # One-hot encode
#         df = pd.get_dummies(df)

#         # Align with training columns
#         df = df.reindex(columns=model_columns, fill_value=0)

#         # Predict probability
#         prediction = model.predict_proba(df)[0][1]
#         probability = round(prediction * 100, 2)

#         # Risk classification
#         if probability > 70:
#             risk = "High Risk"
#             color = "danger"
#         elif probability > 40:
#             risk = "Medium Risk"
#             color = "warning"
#         else:
#             risk = "Low Risk"
#             color = "success"

#         return render_template(
#             "index.html",
#             prediction=probability,
#             risk=risk,
#             color=color
#         )

#     except Exception as e:
#         return f"Error occurred: {str(e)}"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])

        # Proper type conversion
        df['tenure'] = df['tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

        # One-hot encoding
        df = pd.get_dummies(df)

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict_proba(df)[0][1]
        probability = round(prediction * 100, 2)

        if probability > 70:
            risk = "High Risk"
            color = "danger"
        elif probability > 40:
            risk = "Medium Risk"
            color = "warning"
        else:
            risk = "Low Risk"
            color = "success"

        return render_template(
            "index.html",
            prediction=probability,
            risk=risk,
            color=color
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)