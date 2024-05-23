import docker
import os
import json

os.environ["DOCKER_HOSTS"] = '["tcp://localhost:2375", "tcp://127.0.0.1:2375"]'

def main():

    for host in json.loads(os.environ["DOCKER_HOSTS"]):
        print("Connecting to host: ", host)
        client = docker.from_env(environment={
            'DOCKER_HOST': host,
        })

        print(client.containers.list())

    pass


if __name__ == "__main__":
    main()