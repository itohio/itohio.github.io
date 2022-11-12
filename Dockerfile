FROM golang:1.19 as backend
COPY go.mod go.sum /src/
WORKDIR /src
RUN go mod download
COPY cmd /src/cmd
COPY pkg /src/pkg
RUN go build -ldflags="-w -s" ./cmd/app

FROM scratch
COPY --from=backend /src/app /backend
ENTRYPOINT ["/backend"]