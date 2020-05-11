import os

data_path = os.path.join(os.getcwd(), 'trip.csv')
preprocessed_data_path = os.path.join(os.getcwd(), 'Ptrip.csv')
model_weights = os.path.join(os.getcwd(), 'weights/model.h5')
scalar_weights = os.path.join(os.getcwd(), 'weights/scalar.pickle')
encoder_weights = os.path.join(os.getcwd(), 'weights/encoder.pickle')

resample_data_size = 1000

n_features = 10
dense1 = 64
dense2 = 64
dense3 = 64
keep_prob = 0.3
batch_size = 2
num_epoches = 10
validation_split = 0.2