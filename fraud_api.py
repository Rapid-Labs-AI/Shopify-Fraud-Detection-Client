from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load model
model_data = joblib.load('fraud_risk_model_oversampled.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']
risk_encoder = model_data['risk_encoder']
features = model_data['features']

# Shopify API credentials
SHOPIFY_API_KEY = "bf520678d939baaf977cf4fbc5a00ba1"
SHOPIFY_PASSWORD = "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"
SHOPIFY_STORE = "nerdused.myshopify.com"

@app.route('/')
def home():
    return "✅ Fraud Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        df = pd.DataFrame([input_data])
        df["Net Amount"] = df["Total"] - df["Discount Amount"] - df["Taxes"]
        df["Free Shipping"] = (df["Shipping"] == 0).astype(int)

        # Encode categorical fields
        for col in ["Payment Method", "Currency"]:
            df[col] = df[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
            df[col] = label_encoders[col].transform(df[col])

        prediction = model.predict(df[features])[0]
        label = risk_encoder.inverse_transform([prediction])[0]

        return jsonify({"fraud_risk": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/shopify_webhook', methods=['POST'])
def shopify_webhook():
    try:
        order = request.json
        input_data = {
            "Total": float(order.get("total_price", 0)),
            "Shipping": float(order.get("shipping_lines", [{}])[0].get("price", 0)),
            "Taxes": float(order.get("total_tax", 0)),
            "Discount Amount": float(order.get("total_discounts", 0)),
            "Payment Method": order.get("payment_gateway_names", ["Unknown"])[0],
            "Currency": order.get("currency", "USD")
        }
        df = pd.DataFrame([input_data])
        df["Net Amount"] = df["Total"] - df["Discount Amount"] - df["Taxes"]
        df["Free Shipping"] = (df["Shipping"] == 0).astype(int)
        for col in ["Payment Method", "Currency"]:
            known_classes = label_encoders[col].classes_
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
            df[col] = label_encoders[col].transform(df[col])
        prediction = model.predict(df[features])[0]
        risk_label = risk_encoder.inverse_transform([prediction])[0]

        print(f"🛡️ Predicted Fraud Risk: {risk_label}")  # <-- Add this line

        tag_order(order["id"], risk_label)
        return jsonify({"status": "success", "fraud_risk": risk_label})
    except Exception as e:
        print(f"❌ Error: {e}")  # <-- Add this line
        return jsonify({"error": str(e)}), 400


def tag_order(order_id, fraud_label):
    get_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_PASSWORD}@{SHOPIFY_STORE}/admin/api/2023-10/orders/{order_id}.json"
    response = requests.get(get_url)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch order {order_id}")
        return

    existing_tags = response.json().get("order", {}).get("tags", "")
    updated_tags = existing_tags + f",Fraud-{fraud_label}"

    put_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_PASSWORD}@{SHOPIFY_STORE}/admin/api/2023-10/orders/{order_id}.json"
    headers = { "Content-Type": "application/json" }
    data = { "order": { "id": order_id, "tags": updated_tags.strip(',') } }

    put_response = requests.put(put_url, json=data, headers=headers)
    if put_response.status_code != 200:
        print(f"❌ Failed to update tags for order {order_id}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
