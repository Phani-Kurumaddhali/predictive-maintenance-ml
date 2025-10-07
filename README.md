# 🛠️ Predictive Maintenance using Machine Learning

This project demonstrates how Machine Learning can be used to predict equipment maintenance needs based on operational and sensor data.
It contains a synthetic dataset, example preprocessing, and a starter ML training script.

---

### 🎯 Objective
To use data-driven insights to **predict potential equipment failures** and schedule maintenance proactively — improving uptime and reducing costs.

---

### 🧰 Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

### 📁 Structure
```
predictive-maintenance-ml/
│
├── data/
│   ├── sample_data.csv          # Simulated dataset (ready to use)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│
├── src/
│   ├── generate_sample_data.py
│   ├── data_preparation.py
│   ├── model.py
│   ├── predict.py
│
├── requirements.txt
│
└── README.md
```

---

### 🚀 How to run

```bash
# 1. Unzip and open the folder in VS Code or Jupyter
# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Generate dataset (optional, dataset already included)
python src/generate_sample_data.py --output data/sample_data.csv --n 500

# 4. Explore in notebook
jupyter notebook notebooks/01_data_preprocessing.ipynb

# 5. Train model (example)
python src/model.py --input data/sample_data.csv --model_out model/rf_model.joblib
```

---

### 📫 Author
**Phanikumar Kurumaddhali**  
📍 Hyderabad, India
