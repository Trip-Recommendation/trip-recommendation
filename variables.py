import os

data_path = os.path.join(os.getcwd(), 'trip1.csv')

hotel_dict_path = os.path.join(os.getcwd(), 'weights/hotel_dict.h5')
model_weights = os.path.join(os.getcwd(), 'weights/model.h5')
scalar_weights = os.path.join(os.getcwd(), 'weights/scalar.pickle')
encoder_weights = os.path.join(os.getcwd(), 'weights/encoder.pickle')
label_encoder_weights = os.path.join(os.getcwd(), 'weights/label_encoder.pickle')

resample_data_size = 1000
n_recommendation = 5

n_features = 10
dense1 = 64
dense2 = 64
dense3 = 64
keep_prob = 0.3
batch_size = 2
num_epoches = 40
validation_split = 0.2