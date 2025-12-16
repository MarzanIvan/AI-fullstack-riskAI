FROM php:8.3-apache
WORKDIR /var/www/html
RUN apt-get update -y && apt-get install -y libmariadb-dev
RUN docker-php-ext-install mysqli pdo pdo_mysql
COPY ./App /var/www/html/

EXPOSE 80