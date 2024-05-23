import docker
import json
import os
import subprocess

def get_clients():
    clients = [] 
    for host in json.loads(os.environ["NODES"]):
        host = f"tcp://{host}:2375"
        print("Connecting to host: ", host)
        client = docker.from_env(environment={
            'DOCKER_HOST': host,
        })
        clients.append(client)
    return clients


def get_active_jobs():
    for client in get_clients():
        containers = client.containers.list()
        print(containers)   
        active_jobs = []
        for container in containers:
            print(container.name)
            if "ai-training_tensorflow-app" in container.name:
                active_jobs.append(container.name)
    return active_jobs


def start_training():
    for index, host in enumerate(json.loads(os.environ["NODES"])):
        print(f"Starting training on host: {host}")
        command_prefix = ["sshpass", "-p", json.loads(os.environ["NODE_PASS"])[0], "ssh", "-o", "StrictHostKeyChecking=no", f"docker_user@{host}"]

        command = command_prefix + ["cd", "/home/nils/programing/ai-training/", "&&", "docker-compose", "ps"]        
        
        print("COMMAND: ", " ".join(command + ["\n"]), end=" ")

        print(subprocess.check_output(command))