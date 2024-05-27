# image: ray-data-mining

FROM ubuntu:22.04

USER root

RUN set -ex; 
RUN apt-get update
RUN    apt-get install -y sudo python3 python3-pip
RUN    pip3 cache purge
RUN    pip3 install numpy 
RUN    pip3 install scikit-learn 
RUN    pip3 install scipy 
RUN    pip3 install pandas 
RUN    pip3 install matplotlib 
RUN    pip3 install ray[air] 
RUN    pip3 install py_spy
RUN    rm -rf /var/lib/apt/lists/*