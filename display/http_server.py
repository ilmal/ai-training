from flask import Flask, jsonify
import subprocess
from flask_cors import CORS, cross_origin
import json
import os
import pandas as pd
import docker
docker_client = docker.from_env()  

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/docker', methods=['GET'])
def get_docker():
    result = subprocess.check_output(['docker', 'ps']).decode('utf-8')
    return jsonify({'docker': result.strip()})

@app.route('/api/active_jobs', methods=['GET'])
def get_active_jobs():
    
    result = []
    for host in json.loads(os.environ["NODES"]):
        print("Connecting to host: ", host)
        client = docker.from_env(environment={
            'DOCKER_HOST': host,
        })

        
        result += client.containers.list()

    return_arr = []
    for container in result:
        return_arr.append({
            'name': container.name,
            'status': container.status,
            'image': container.image.tags[0],
            'id': container.id,
        })

    return jsonify({'data': json.dumps(return_arr)})


@app.route('/api/past_jobs', methods=['GET'])
def get_past_jobs():

    # get all finished jobs
    df = pd.read_csv(os.environ["RESULTS_DIR"] + "model_data.csv", index_col=0)

    finished_models = df[df['training_finished']]['model_generation']

    return_arr = []

    for finished_model_number in finished_models:
        if str(finished_model_number)[0] != "0":
            finished_model_number = "0" + str(finished_model_number)

        model_logs_path = os.environ["RESULTS_DIR"] + "model_logs/model_" + finished_model_number + ".txt"
        if not os.path.exists(model_logs_path):
            raise Exception("Model logs not found for model " + finished_model_number)

        model_logs = open(model_logs_path, "r").read()

        model_code_path = os.environ["MODELS_DIR"] + "/model_" + finished_model_number + ".py"
        if not os.path.exists(model_code_path):
            raise Exception("Model code not found for model " + finished_model_number)

        model_code = open(model_code_path, "r").read()

        # print("Train acc: ", model_logs[-54:-46])
        # print("Train loss: ", model_logs[-75:-65])
        # print("epoch: ", model_logs[-89:-84])

        return_arr.append({
            'model': finished_model_number,
            'accuracy': model_logs[-9:],
            'loss': model_logs[-34:-24],
            'train_acc': model_logs[-54:-46],
            'train_loss': model_logs[-75:-65],
            'epoch': model_logs[-89:-84],
            'logs': model_logs,
            'model_code': model_code
        })

    return jsonify({'data': json.dumps(return_arr, indent=6)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="3001")
