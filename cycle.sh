#!/bin/bash

# Function to wait for a Docker container to complete
wait_for_container() {
    container_name=$1

    while [ "$(sudo docker inspect -f '{{.State.Running}}' $container_name 2>/dev/null)" == "true" ]; do
        sleep 1
    done
}

# # Run the first container
# sudo docker-compose run tensorflow-app

# # Wait for the first container to complete
# wait_for_container "tensorflow-app"

# # Run the second container
# sudo docker-compose run generate_new_model

# # Wait for the second container to complete
# wait_for_container "generate_new_model"

# Repeat the process (looping)
while true; do
    # Run the first container again
    sudo docker-compose run tensorflow-app

    # Wait for the first container to complete
    wait_for_container "tensorflow-app"

    # Run the second container again
    sudo docker-compose run generate_new_model

    # Wait for the second container to complete
    wait_for_container "generate_new_model"
done