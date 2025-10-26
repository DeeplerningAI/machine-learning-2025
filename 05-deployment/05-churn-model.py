import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



# data preprocessing


df = pd.read_csv('data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df['churn'] = (df['churn'] == 'yes').astype(int)


# In[15]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[16]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[36]:


def train(df_train, y_train, C=1.0):
    dicts_train = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts_train)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)

    return dv, model


# In[37]:


def predict(df, dv, model):
    dicts_predict = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts_predict)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[49]:


C = 1.0
n_splits = 5


# In[50]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]    

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[53]:


auc


# In[54]:


scores


# In[66]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# save the model

# In[67]:


import pickle


# In[68]:


output_file = f'model_C={C}.bin'
output_file


# In[69]:


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# In[70]:


with open(output_file, 'wb') as f_out: # 'wb' means write-binary
    pickle.dump((dv, model), f_out)


# load the model

# In[71]:


import pickle


# In[72]:


model_file = 'model_C=1.0.bin'


# In[73]:


with open(output_file, 'rb') as f_in: # 'wb' means write-binary
    dv, model = pickle.load(f_in)


# In[74]:


dv, model


# In[75]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'ds1',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month_to_month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electric_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[78]:


X = dv.transform([customer])


# In[79]:


X


# In[80]:


model.predict_proba(X)


# In[81]:


model.predict_proba(X)[0, 1]


# In[ ]:




