import os
import json
import pandas as pd

from variables import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').disabled = True

from main import TripRecommendation

from flask import Flask
from flask import jsonify
from flask import request
from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()

'''
        python -W ignore app.py

'''

app = Flask(__name__)

model = TripRecommendation()
model.run()

def train_task():
    model = TripRecommendation()
    model.run_finetune()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    user_id = message['user_id']
    hotels = model.prediction(user_id)

    response = {
            'hotels': hotels
    }
    return jsonify(response)

if __name__ == "__main__":
    scheduler.add_job(func=train_task, trigger="interval", seconds=learning_interval)
    scheduler.start()
    app.run(debug=True, host=host, port= port, threaded=False)
