FROM rocm/tensorflow:latest

# Set the working directory
WORKDIR /app

# Copy your TensorFlow application files to the container
COPY . /app

# install pip3 dependencies
RUN python3 -m pip install -r requirements.txt
