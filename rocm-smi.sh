#!/bin/bash

container_prefix="ai-training_tensorflow-app"
cmd="sudo docker exec -it"

# Get a list of container IDs with the specified prefix
container_ids=$(sudo docker ps -aq --filter "name=${container_prefix}")

for container_id in $container_ids; do
    container_name=$(sudo docker inspect --format='{{.Name}}' "$container_id" | cut -c 2-)

    # Run the rocm-smi command in each container
    for i in $(seq 50000); do
        $cmd "$container_name" rocm-smi
        sleep 1
    done
done
