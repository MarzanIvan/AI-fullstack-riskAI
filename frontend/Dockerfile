FROM node:alpine AS build


COPY package.json package.json
COPY /src /src
RUN rm -rf node_modules
RUN rm -rf node_modules package-lock.json
#RUN npm install
COPY . .

#RUN npm run build

FROM nginx:stable-alpine

COPY --from=build /dist /usr/share/nginx/html
RUN mkdir /usr/share/nginx/html/src
COPY --from=build /src /usr/share/nginx/html/src
COPY --from=build nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD [ "nginx", "-g", "daemon off;"]
