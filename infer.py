from gnn.estimator_fns import *
from gnn.graph_utils import *
from gnn.data import *
from gnn.utils import *
from gnn.pytorch_model import *
import pickle

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

columns_to_select = ['Year', 'Day of Week', 'Hour', 'Amount', 'Use Chip', 'Merchant Name', 'MCC', 'Is Fraud?']

data = {
    'Year': [2002],
    'Day of Week': [5],        
    'Hour': [14],             
    'Amount': [134.09],
    'Use Chip': [1],          
    'Merchant Name': [3527213246127876953],
    'MCC': [5300],           
    'Is Fraud?': [0]        
}

df_new = pd.DataFrame(data)
X_new = df_new[columns_to_select[:-1]]
pred = model.predict(X_new)
print("Prediction:", pred)