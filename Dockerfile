FROM ubuntu:14.04
MAINTAINER Ethan Goan <e.goan@student.qut.edu.au>


#install all of the Python Packages required

RUN apt-get update
RUN apt-get install wget

#install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
RUN bash Anaconda2-4.3.0-Linux-x86_64.sh -p /anaconda -p
RUN rm Anaconda2-4.3.0-Linux-x86_64.sh
ENV PATH=/anaconda/bin:${PATH}
RUN conda update -y conda

##############
#
#install CUDA
#SOURCED FROM https://gitlab.com/nvidia/cuda/blob/ubuntu14.04/8.0/runtime/Dockerfile
#
#https://hub.docker.com/r/nvidia/cuda/
#
##############

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list
ENV CUDA_VERSION 8.0
LABEL com.nvidia.cuda.version="8.0"
ENV CUDA_PKG_VERSION 8-0=8.0.61-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvrtc-$CUDA_PKG_VERSION \
        cuda-nvgraph-$CUDA_PKG_VERSION \
        cuda-cusolver-$CUDA_PKG_VERSION \
        cuda-cublas-$CUDA_PKG_VERSION \
        cuda-cufft-$CUDA_PKG_VERSION \
        cuda-curand-$CUDA_PKG_VERSION \
        cuda-cusparse-$CUDA_PKG_VERSION \
        cuda-npp-$CUDA_PKG_VERSION \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-$CUDA_VERSION /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64





##############
#
#installing all the python packages required
#Some will already be installed with Anaconda, justdouble checking
#
##############


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

##############
#
# Copy the GPU-LIBSVM and the normal LIBSVM source over
# GPU enhanced version is used for training
#
##############

COPY CUDA ./CUDA/
COPY libsvm ./LIBSVM/

RUN make clean -C ./CUDA/
RUN make -C ./CUDA/
RUN make clean -C ./LIBSVM/
RUN make -C ./LIBSVM/

COPY preprocess.sh .

