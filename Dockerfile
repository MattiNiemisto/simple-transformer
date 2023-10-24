FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update -y && apt install -y wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN unlink /usr/bin/python3
RUN apt install -y -qq python3.9 && ln -s /usr/bin/python3.9 /usr/bin/python3
RUN apt -f -y install
RUN apt update -y && apt dist-upgrade -y
RUN apt install -y python3-pip python3.9-venv 
RUN apt upgrade -y

WORKDIR workspace
COPY requirements.txt .

RUN ["bash", "-c", "python3 -m venv env"]
RUN ["bash", "-c", "source env/bin/activate"]
RUN ["bash", "-c", "pip3 install --upgrade pip"]
RUN ["bash", "-c", "pip3 install -r requirements.txt" ] 

COPY src/ .

CMD "bash"