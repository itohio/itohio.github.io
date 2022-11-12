FROM golang:1.19 as backend
COPY go.mod go.sum /src/
WORKDIR /src
RUN go mod download
COPY ./cmd ./pkg /src/
RUN go build -w -s ./...

FROM scratch
COPY --from=backend /src/backend /backend
ENTRYPOINT ["/backend"]