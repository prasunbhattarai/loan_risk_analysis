import json
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor  
import shap

with open("src/model/analysis/describe.json", "r") as f:
    columns = json.load(f)

def load_model():
    model_path = "src/model/core/xgboostmodel.json"

    model = XGBRegressor()   
    model.load_model(model_path)

    return model


def shap_explainer(model, X, features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]

    shap_df = pd.DataFrame({
        "feature": features,
        "impact": shap_values
    })

    shap_df["abs_impact"] = shap_df["impact"].abs()

    top_features = shap_df.sort_values("abs_impact", ascending=False).head(5)
    top_features["direction"] = top_features["impact"].apply(
        lambda x: "increase" if x > 0 else "decrease"
    )
    return top_features[["feature", "impact", "direction"]]


def get_feature_details(top_features, describe_data, original_data):
    enriched = []

    for feat in top_features:
        name = feat["feature"]

        if name not in describe_data:
            continue

        stats = describe_data[name]

        enriched.append({
            "feature": name,
            "impact": feat["impact"],
            "direction": feat["direction"],
            "value": original_data[name],  
            "mean": stats["mean"],
            "min": stats["min"],
            "max": stats["max"],
            "25%": stats["25%"],
            "50%": stats["50%"],
            "75%": stats["75%"]
        })

    return enriched


def format_features_for_llm(features):
    text = ""

    for f in features:
        value = f["value"]
        mean = f["mean"]

        is_binary = f["max"] == 1.0 and f["min"] == 0.0

        if is_binary:
            if value == 1:
                description = "This condition is present"
            else:
                description = "This condition is not present"

            text += f"""
            Feature: {f['feature']}
            - {description}
            - Most applicants do NOT have this condition
            - Effect on risk: tends to {f['direction']} risk

            """

        else:
            deviation = "higher" if value > mean else "lower"

            text += f"""
Feature: {f['feature']}
- Applicant value: {value}
- Typical range: {f['25%']} to {f['75%']}
- Compared to average: {deviation} than typical
- Effect on risk: tends to {f['direction']} risk

"""

    return text.strip()




def predict(data):
    original_data = data.iloc[0].to_dict()
    model = load_model()

    string_df = data.select_dtypes(include=['object']).columns.tolist()
    features = joblib.load("src/model/preprocessing/features.pkl")
    data = data[features]

    scaler = joblib.load("src/model/preprocessing/scaler.pkl")
    encoders = joblib.load("src/model/preprocessing/encoder.pkl")

    for col in string_df:
        data[col] = encoders[col].transform(data[col])
    X = scaler.transform(data)
    prediction = model.predict(X)
    top_features = shap_explainer(model, X, features)
    # print(top_features)
    enriched_features = get_feature_details(
        top_features.to_dict(orient="records"),
        columns,
        original_data
    )
    # print(enriched_features)
    text_for_llm = format_features_for_llm(enriched_features)
    # print(text_for_llm)
    return prediction[0], text_for_llm

# predict("prediction_data.csv")