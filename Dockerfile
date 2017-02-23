FROM nvidia/cuda:8.0-runtime-ubuntu14.04
MAINTAINER Ethan Goan <e.goan@student.qut.edu.au>


#install all of the Python Packages required

RUN apt-get update
RUN apt-get install wget -y

#install Anaconda
RUN wget https://repo.continuum.io/archive/Anaconda2-4.3.0-Linux-x86_64.sh
RUN bash Anaconda2-4.3.0-Linux-x86_64.sh -p /anaconda -b
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
#############
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-core-$CUDA_PKG_VERSION \
    cuda-misc-headers-$CUDA_PKG_VERSION \
    cuda-command-line-tools-$CUDA_PKG_VERSION \
    cuda-nvrtc-dev-$CUDA_PKG_VERSION \
    cuda-nvml-dev-$CUDA_PKG_VERSION \
    cuda-nvgraph-dev-$CUDA_PKG_VERSION \
    cuda-cusolver-dev-$CUDA_PKG_VERSION \
    cuda-cublas-dev-$CUDA_PKG_VERSION \
    cuda-cufft-dev-$CUDA_PKG_VERSION \
    cuda-curand-dev-$CUDA_PKG_VERSION \
    cuda-cusparse-dev-$CUDA_PKG_VERSION \
    cuda-npp-dev-$CUDA_PKG_VERSION \
    cuda-cudart-dev-$CUDA_PKG_VERSION \
    cuda-driver-dev-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}


##############
#
#installing all the python packages required
#Some will already be installed with Anaconda, justdouble checking
#
##############

RUN pip install --upgrade pip
RUN pip install pydicom
#RUN pip install numpy
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
COPY breast_cython.pyx .
COPY arguments.py .
COPY log.py .

#copy script to compile Cython Files
COPY compile.sh .
#make it executable
RUN chmod u+x ./compile.sh
#give 755 permissions
RUN chmod 755 compile.sh
#now run it
RUN ./compile.sh

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

