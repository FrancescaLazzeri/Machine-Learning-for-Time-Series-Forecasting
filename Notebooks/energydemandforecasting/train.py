#%% [markdown]
# ## Training script
# This script will be given to the Estimator which is configured in the AML training script.
# It is parameterized for training on `energy.csv` data.

#%% [markdown]
# ### Import packages.
# utils.py needs to be in the same directory as this script, i.e., in the source directory `energydemandforcasting`.

#%%
import argparse
import os
import numpy as np
import pandas as pd
import azureml.data
import pickle

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, mape
from azureml.core import Run

#%% [markdown]
# ### Parameters
# * COLUMN_OF_INTEREST: The column containing data that will be forecasted.
# * NUMBER_OF_TRAINING_SAMPLES: The number of training samples that will be trained on.
# * ORDER: A tuple of three non-negative integers specifying the parameters p, d, q of an Arima(p,d,q) model, where:
#      * p: number of time lags in autoregressive model,
#      * d: the degree of differencing,
#      * q: order of the moving avarage model.
# * SEASONAL_ORDER: A tuple of four non-negative integers where the first three numbers
#      specify P, D, Q of the Arima terms of the seasonal component, as in ARIMA(p,d,q)(P,D,Q).
#      The fourth integer specifies m, i.e, the number of periods in each season.

#%%
COLUMN_OF_INTEREST = "load"
NUMBER_OF_TRAINING_SAMPLES = 2500
ORDER = (4, 1, 0)
SEASONAL_ORDER = (1, 1, 0, 24)

#%% [markdown]
# ### Import script arguments
# Here, Azure will read in the parameters, specified in the AML training.

#%%
parser = argparse.ArgumentParser(description="Process input arguments")
parser.add_argument("--data-folder", default="./data/", type=str, dest="data_folder")
parser.add_argument("--filename", default="energy.csv", type=str, dest="filename")
parser.add_argument("--output", default="outputs", type=str, dest="output")
args = parser.parse_args()
data_folder = args.data_folder
filename = args.filename
output = args.output
print("output", output)
#%% [markdown]
# ### Prepare data for training
# * Import data as pandas dataframe
# * Set index to datetime
# * Specify the part of the data that the model will be fitted on
# * Scale the data to the interval [0, 1]

#%%
# Import data
energy = load_data(os.path.join(data_folder, filename))
# As we are dealing with time series, the index can be set to datetime.
energy.index = pd.to_datetime(energy.index, infer_datetime_format=True)

# Specify the part of the data that the model will be fitted on.
train = energy.iloc[0:NUMBER_OF_TRAINING_SAMPLES, :]

# Scale the data to the interval [0, 1].
scaler = MinMaxScaler()
train[COLUMN_OF_INTEREST] = scaler.fit_transform(
    np.array(train.loc[:, COLUMN_OF_INTEREST].values).reshape(-1, 1)
)
#%% [markdown]
# ### Fit the model

#%%
model = SARIMAX(
    endog=train[COLUMN_OF_INTEREST].tolist(), order=ORDER, seasonal_order=SEASONAL_ORDER
)
model.fit()

#%% [markdown]
# ### Save the model
# The model will be saved on Azure in the specified directory as a pickle file.

#%%
# Create a directory on Azure in which the model will be saved.
os.makedirs(output, exist_ok=True)

# Write the the model as a .pkl file to the specified directory on Azure.
with open(output + "/arimamodel.pkl", "wb") as m:
    pickle.dump(model, m)

# with open('arimamodel.pkl', 'wb') as m:
#     pickle.dump(model, m)

#%%
