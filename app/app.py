from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load mô hình và các đối tượng đã lưu
duongdan_models = os.path.join(os.path.dirname(__file__), '../models')
model_ridge = joblib.load(os.path.join(duongdan_models, 'ridge_model.pkl'))
model_rf = joblib.load(os.path.join(duongdan_models, 'rf_model.pkl'))
scaler = joblib.load(os.path.join(duongdan_models, 'scaler.pkl'))
imputer = joblib.load(os.path.join(duongdan_models, 'imputer.pkl'))
feature_names = joblib.load(os.path.join(duongdan_models, 'feature_names.pkl'))

@app.route('/', methods=['GET', 'POST'])
def index():
    ket_qua = None
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form và tạo DataFrame có đúng tên cột
            du_lieu_dict = {ten: [float(request.form.get(ten))] for ten in feature_names}
            dau_vao_df = pd.DataFrame(du_lieu_dict)

            # Chọn mô hình
            model_chon = request.form.get('model')
            if model_chon == 'ridge':
                # Đảm bảo đầu vào đúng cột
                dau_vao_scaled = pd.DataFrame(
                    scaler.transform(dau_vao_df), columns=feature_names
                )
                dau_vao_imputed = pd.DataFrame(
                    imputer.transform(dau_vao_scaled), columns=feature_names
                )
                ket_qua = float(model_ridge.predict(dau_vao_imputed)[0])
            else:
                # Random Forest không cần scaler hay imputer
                ket_qua = float(model_rf.predict(dau_vao_df)[0])

        except Exception as e:
            ket_qua = f"Lỗi: {e}"

    return render_template('index.html', feature_names=feature_names, ket_qua=ket_qua)


if __name__ == '__main__':
    app.run(debug=True)
