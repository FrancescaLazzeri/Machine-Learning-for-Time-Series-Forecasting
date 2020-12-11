#%% [markdown]

# ## score.py
# This script allows predicting the rest of the energy time series with the registered model. It will be given to the container image.
# It must contain an `init()` and a `run()` function.
# For more information about the `score.py` file in general, please refer to the [link](https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-deploy-models-with-aml#deploy-as-a-web-service).

# #### This script is parameterized for a json object containing the `energy.csv` data and the registred model `arimamodel`.
# * If you use another model in the deployment, you have to adjust the variable `MODEL_NAME`.
# * If you train your model on other data, you have to adjust it in the deployment and in this file in `DATA_NAME`.
# * The name of the column containing the trained data must be the same in this file. You can specify it in `DATA_COLUMN_NAME`.
# In our case the name is `load`.
# * The number of training samples used in the training of your model must also be specified in this file.
# In our case, we used the 2500 first data points of the series for training, therefore, `NUMBER_OF_TRAINING_SAMPLES` also needs to be set to `2500`.

#%% [markdown]
# ### Import packages

#%%
import pickle
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from azureml.core.model import Model

#%% [markdown]
# ### Setup of the main variables
# * MODEL_NAME: The name of the model which has been given during model registering.
# * DATA_COLUMN_NAME: The column containing data that will be forecasted.
# * DATA_NAME: The name of the file containing the data, as specified in the deployment script.
# * NUMBER_OF_TRAINING_SAMPLES: The number of training samples.
# * HORIZON: The number of time steps to be forecasted.

#%%
MODEL_NAME = "arimamodel"
DATA_NAME = "energy"
DATA_COLUMN_NAME = "load"
NUMBER_OF_TRAINING_SAMPLES = 2500
HORIZON = 10

#%% [markdown]
# ### Init function
# The function is executed once the Docker container is started.
# It loads the model into a global object.

#%%
def init():
    global model
    model_path = Model.get_model_path(MODEL_NAME)
    # deserialize the model file back into a sklearn model
    with open(model_path, "rb") as m:
        model = pickle.load(m)


#%% [markdown]
# ### Run function
# The function uses the model to predict a value based on the time series.

#%%
def run(energy):
    try:
        # load data as pandas dataframe from the json object.
        energy = pd.DataFrame(json.loads(energy)[DATA_NAME])
        # take the training samples
        energy = energy.iloc[0:NUMBER_OF_TRAINING_SAMPLES, :]

        scaler = MinMaxScaler()
        energy[DATA_COLUMN_NAME] = scaler.fit_transform(energy[[DATA_COLUMN_NAME]])
        model_fit = model.fit()

        prediction = model_fit.forecast(steps=HORIZON)
        prediction = pd.Series.to_json(pd.DataFrame(prediction), date_format="iso")

        # you can return any data type as long as it is JSON-serializable
        return prediction
    except Exception as e:
        error = str(e)
        return error
