FROM ubuntu:14.04
MAINTAINER Ethan Goan <e.goan@student.qut.edu.au>


#install all of the Python Packages required

# Install dependencies for running generate_labels.py



#RUN apt-get install --reinstall software-properties-common
#RUN apt-get -qq -y install software-properties-common
#RUN apt-add-repository universe#
#RUN apt-get -qq update
#RUN apt-get -qq -y install python-pip


RUN apt-get update


FROM python
RUN pip install --upgrade pip
RUN pip install pydicom
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install scipy
RUN pip install scikit-image
RUN pip install PyWavelets
RUN pip install scikit-learn
RUN pip install entropy




COPY main_thread.py .
COPY read_files.py .
COPY feature_extract.py .
COPY my_thread.py .
COPY breast.py  .
COPY arguments.py .
COPY log.py .

COPY preprocess.sh .

