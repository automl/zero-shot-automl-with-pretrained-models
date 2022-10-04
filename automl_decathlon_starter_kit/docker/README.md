# Docker image for AutoML Decathlon competition

The Docker image used for AutoML Decathlon challenge is created with `Dockerfile`
in this directory.

Here is the [link](https://hub.docker.com/r/automldec/decathlon) to Docker Hub.

## Build Docker image for GPU/CPU
Edit the `Dockerfile` to install additional libraries, then run the following command to build the image
```
docker build -t automldec/decathlon:latest .
```
If you have push access, you can do
```
docker push automldec/decathlon:latest
```
