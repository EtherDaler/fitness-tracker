FROM python:3.10-slim

RUN sed -i 's#http://deb.debian.org/#http://ftp.de.debian.org/#' /etc/apt/sources.list.d/debian.sources
RUN apt-get update --fix-missing && apt-get install -y vim --fix-missing

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY ./app /app/app
COPY requirements.txt /app/

RUN sysctl -w net.ipv6.conf.all.disable_ipv6=1

RUN python -m pip install --upgrade pip
RUN pip install --default-timeout=100 -r requirements.txt 

EXPOSE 8000

# the inner logic saves the active workout connection in RAM, so we need to run only one worker
# since there arises a case when a connection is saved in `A` worker and the request is handled by `B` worker raising errors
CMD ["./scripts/wait-for-it.sh", "db:5432", "--", "gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "-w", "1"]
