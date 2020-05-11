import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from variables import*

# create a folder named 'data' in the project folder

def get_data():
    df = pd.read_csv(data_path, index_col=False)

    labels = df['TRIP_ID'].values
    onehot_data = onehotEncode_ages(df)
    numeric_data = normalize_data(df)

    Inputs = np.hstack((numeric_data,onehot_data))

    return Inputs, labels

def encode_destination(df):
    destination = df['DESTINATION'].values
    if not os.path.exists(encoder_weights):
        encoder = LabelEncoder()
        encoder.fit(destination)
        joblib.dump(encoder, encoder_weights)

    encoder = joblib.load(encoder_weights)
    encode_des = encoder.transform(destination)
    return encode_des

def onehotEncode_ages(df):
    bins = [0, 20, 30, 40, 50, np.inf]
    age_limits = df['AGE_LIMIT'].values

    N = len(age_limits)
    K = len(bins) - 1
    onehot_ages = np.zeros((N, K), dtype=np.int32)
    for n, age_limit in enumerate(age_limits):
        ages = age_limit.split('-')
        ub = int(ages[1])
        lb = int(ages[0])
        inds = np.digitize([lb, ub], bins, right=True)
        for k in inds:
            onehot_ages[n,k] = 1
    return onehot_ages

def normalize_data(df):
    encode_des = encode_destination(df)
    encode_des = encode_des.reshape(len(encode_des),1)

    data = df[['NUMBER_OF_PEOPLE', 'DAYS', 'PER_PERSON_BUDGET', 'TOTAL_BUDGET']].values
    data = data.reshape(len(data),4)

    numeric_data = np.hstack((data,encode_des))

    if not os.path.exists(scalar_weights):
        scaler = StandardScaler()
        scaler.fit(numeric_data)
        joblib.dump(scaler, scalar_weights)

    scaler = joblib.load(scalar_weights)
    numeric_data = scaler.transform(numeric_data)
    return numeric_data
