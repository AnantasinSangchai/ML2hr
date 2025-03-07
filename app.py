import os
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับค่าจากฟอร์ม
        complain = int(request.form['complain'])
        age = int(request.form['age'])
        is_active = int(request.form['is_active'])
        num_products = int(request.form['num_products'])
        geography = request.form['geography']
        balance = float(request.form['balance'])

        # แปลงค่าหมวดหมู่เป็นตัวเลข
        geo_map = {"France": 0, "Germany": 1, "Spain": 2}
        geography = geo_map.get(geography, 0)

        # สร้าง DataFrame สำหรับโมเดล
        input_df = pd.DataFrame([[complain, age, is_active, num_products, geography, balance]], 
                                columns=["Complain", "Age", "IsActiveMember", "NumOfProducts", "Geography", "Balance"])

        # โหลดโมเดลและ scaler
        with open("customer_churn_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        # ปรับขนาดข้อมูล
        input_data = scaler.transform(input_df)

        # ทำนายผล
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1][0]

        # ส่งผลลัพธ์
        if prediction[0] == 1:
            result = f"🚨 ลูกค้ารายนี้มีแนวโน้มที่จะเลิกใช้บริการ ({probability:.2%} ความน่าจะเป็น)"
        else:
            result = f"✅ ลูกค้ารายนี้มีแนวโน้มที่จะอยู่ต่อ ({(1 - probability):.2%} ความน่าจะเป็น)"
        
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# กำหนดให้ Gunicorn ใช้ตัวแปร app
if __name__ == "__main__":
    app.run(debug=True)
