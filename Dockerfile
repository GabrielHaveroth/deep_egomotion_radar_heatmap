FROM tensorflow/tensorflow:latest-gpu
ADD . /developer
USER gabs
ENV HOME /home/gabs
LABEL maintainer="haverothgabriel@gmail.com"