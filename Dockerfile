# Must use a Cuda version 11+
#FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git \
    python3.10 python3-pip

#python3.10-venv
#    build-essential libgl-dev libglib2.0-0

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD xformers/ xformers/
RUN cat xformers/* >xformers-0.0.15.dev0+303e613.d20221128-cp310-cp310-linux_x86_64.whl
RUN pip install xformers-0.0.15.dev0+303e613.d20221128-cp310-cp310-linux_x86_64.whl

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD app.py .

CMD python3 -u server.py
