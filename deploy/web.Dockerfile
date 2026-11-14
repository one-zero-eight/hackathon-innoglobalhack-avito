# Веб-клиент: сборка статики Vite в node-слое, раздача через nginx.
FROM node:22-alpine AS build

WORKDIR /app

# Манифест (и lock-файл, если появится) — отдельным слоем для кэша npm install
COPY src/web/package*.json ./
RUN npm install

# Остальной код клиента (src/node_modules и dist исключены .dockerignore)
COPY src/web ./
RUN npm run build

FROM nginx:alpine

# Конфиг: статика из dist + прокси /api/ на сервис api
COPY deploy/web.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
