FROM rocm/tensorflow:latest

WORKDIR /app

COPY requirements.txt /app

# install pip3 dependencies
RUN python3 -m pip install -r requirements.txt
