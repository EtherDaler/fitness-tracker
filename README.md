# Fitness Tracker Website Monorepo

What is this app?
This is a full stack app that is used to track the user's activity and fitness. A user can create a new workout defined in the server. The user will open their camera and the app will send the live footage in the backend server, where ML models will calculate how accurate the exercise is. The user's activity will be stored, various mathematical calculations will be done on the data and the user will be shown the progress in the frontend in the charts.

## Structure

This is a monorepo for a fitness tracker website containing the full stack app inside the server/ directory, the nginx Dockerfile for the reverse proxy in the nginx/ directory, and some scripts in the /scripts directory.

Docker along with Docker Compose is used to ensure that the app can be run in any environment without any issues in few commands.

## Nginx

Nginx is used as a reverse proxy to route requests to the appropriate service. The nginx/ directory contains the Dockerfile for the nginx image. It also contains init.conf and ssh.conf which are used to configure the nginx server. The init.conf is used to initially setup the server so that certbot can generate SSL certificates for the domain. The ssh.conf is used to configure the server to allow SSH connections.

## Scripts

A very basic script that is used to generate a self-signed SSL certificate for the domain. It is a docker command that setups a certbot docker image and requests for a certificate for the domain. The certificate is then stored in the certbot/ generated directory.

## Server

The main full stacked app made using FastAPI. The app resides inside the [/app](/server/app/) directory. The structure is following.
The backend API is served in the [/api/v1/](/server/app/api/v1/) directory with endpoints corresponding to file names.
The frontend is made in pure HTML, CSS and JavaScript to mimimize the size of the app. The frontend is served in the [/templayes/](/server/app/templates/) directory. All the static content is there in [/static](/server/app/static/) directory. The server is setup using a Dockerfile.

## Running the app

Make sure you have Docker and Docker Compose installed on your system. Then run the following command to start the app. This one command will start your app.

```bash
docker-compose up
```

If your domain doesn't has SSL ocnfigured, then in the docker-compose.yml set the [init.conf](/nginx/init.conf) as the nginx config.

Then generate a SSL certificate using the [bash command](/scripts/certbot-ssl-init.sh). _Edit as required._

After generating the certificate, change the config inside the docker-compose.yml to [ssl.conf](/nginx/ssl.conf) and then run the docker-compose up command again.

```bash
docker-compose up
```
