docker run -it --rm \
-v "$(pwd)/certbot/conf:/etc/letsencrypt" \
-v "$(pwd)/certbot/www:/var/www/certbot" \
certbot/certbot certonly \
--webroot --webroot-path=/var/www/certbot \
-d sirshak.tech -m boharasirshak@gmail.com --agree-tos --no-eff-email
