FROM python:3.7-alpine as base
RUN apk update && apk --no-cache add libpq mariadb-connector-c libjpeg zlib

FROM base as builder
RUN apk add --no-cache linux-headers g++ python3-dev mariadb-dev musl-dev jpeg-dev zlib-dev
COPY app/requirements.txt /requirements.txt
RUN pip wheel --wheel-dir=/root/wheels -r /requirements.txt 

FROM base
COPY --from=builder /root/wheels /root/wheels
ADD app /app
RUN pip3 install --no-index --find-links=/root/wheels -r /app/requirements.txt && rm -r /app/static
EXPOSE 8080
WORKDIR "/app"
CMD ["/usr/local/bin/uwsgi", "--ini", "uwsgi.ini"]
