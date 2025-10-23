import pickle

# Load model và DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Dữ liệu mẫu
datapoint = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

# Chuyển đổi dữ liệu sang dạng vector
X
