#FROM python:3.7
#FROM tensorflow/tensorflow:2.3.2-gpu
FROM tensorflow/tensorflow:2.7.0-gpu

ENV DEBCONF_NOWARNINGS yes
ENV DEBIAN_FRONTEND=noninteractive

ENV TZ="Asia/Tokyo"
RUN apt-get update && apt-get install -y tzdata
RUN apt-get install -y --quiet --no-install-recommends \
  graphviz 
  
# == python modules==
COPY requirements.txt ./
RUN pip install -q --upgrade pip
RUN pip install -r requirements.txt -q

RUN pip install "opencv-python-headless<4.3.0" -q
#RUN pip install opencv-contrib-python
#RUN apt-get install -y libgl1-mesa-dev

# finalize
EXPOSE 8888
WORKDIR /work
CMD ["/bin/bash"]