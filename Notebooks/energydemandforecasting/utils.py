import os
import re
import sys
import zipfile
from collections import UserDict

import numpy as np
import pandas as pd
import requests


def load_data(data_path):
    """Load the GEFCom 2014 energy load data"""

    energy = pd.read_csv(os.path.join(data_path), parse_dates=["timestamp"])

    # Reindex the dataframe such that the dataframe has a record for every time point
    # between the minimum and maximum timestamp in the time series. This helps to
    # identify missing time periods in the data (there are none in this dataset).

    energy.index = energy["timestamp"]
    energy = energy.reindex(
        pd.date_range(min(energy["timestamp"]), max(energy["timestamp"]), freq="H")
    )
    energy = energy.drop("timestamp", axis=1)
    energy = energy.drop("temp", axis=1)

    return energy


def mape(predictions, actuals):
    """Mean absolute percentage error"""
    return ((predictions - actuals).abs() / actuals).mean()
