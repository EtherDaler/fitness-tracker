FROM python:3.10

RUN sed -i 's#http://deb.debian.org/#http://ftp.de.debian.org/#' /etc/apt/sources.list.d/debian.sources
RUN apt-get update --fix-missing && apt-get install -y vim --fix-missing && apt-get install -y traceroute

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get install -y dnsutils

WORKDIR /app

COPY ./app /app/app
COPY requirements.txt /app/

RUN apt-get install -y ca-certificates && apt-get install -y iputils-ping && apt-get install -y openssl

#RUN nslookup pypi.org

#RUN openssl s_client -connect google.com:443

#RUN traceroute pypi.org > /app/traceroute_output.txt

#RUN openssl s_client -connect pypi.tuna.tsinghua.edu.cn:443

#RUN apt-get install -y curl
#RUN curl -I https://pypi.python.org/simple/ --verbose

RUN python -m pip install --upgrade pip --proxy http://root:PkSYV1venHAWSXdO@95.163.230.17:22
RUN pip install --no-cache-dir -r requirements.txt --trusted-host pypi.python.org --proxy http://root:PkSYV1venHAWSXdO@95.163.230.17:22

EXPOSE 8000

# the inner logic saves the active workout connection in RAM, so we need to run only one worker
# since there arises a case when a connection is saved in `A` worker and the request is handled by `B` worker raising errors
CMD ["./scripts/wait-for-it.sh", "db:5432", "--", "gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "-w", "1"]
