import docker


def get_active_jobs():
    client = docker.from_env()
    containers = client.containers.list()
    print(containers)
    active_jobs = []
    for container in containers:
        if "ai-training" in container.name:
            active_jobs.append(container.name)
    return active_jobs


def start_training(IMAGE_NAME):
    client = docker.from_env()
    client.containers.run(IMAGE_NAME, detach=True)