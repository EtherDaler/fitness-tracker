FROM python:3.10

# Edit /etc/default/docker file to add custom DNS server
RUN echo 'DOCKER_OPTS="--dns 172.16.16.10"' >> /etc/default/docker

#working directory where i have placed container 
#on top of container

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY ./app /app/app
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 8000

# the inner logic saves the active workout connection in RAM, so we need to run only one worker
# since there arises a case when a connection is saved in `A` worker and the request is handled by `B` worker raising errors
CMD ["./scripts/wait-for-it.sh", "db:5432", "--", "gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "-w", "1"]
