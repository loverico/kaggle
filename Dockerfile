FROM jupyter/tensorflow-notebook
ARG ROOT_PASSWD
ARG PASSWD
ENV CUDA_VERSION=10.1

# Install CUDA10
USER root
RUN apt update && apt upgrade -y
RUN apt install -y gnupg2
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/\
ubuntu1810/x86_64/cuda-repo-ubuntu1810_10.1.105-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1810_10.1.105-1_amd64.deb
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/\
compute/cuda/repos/ubuntu1810/x86_64/7fa2af80.pub
RUN apt update
#RUN apt install -y cuda
RUN rm cuda-repo-ubuntu1810_10.1.105-1_amd64.deb

# Install PyCUDA
#ENV PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH
#ENV CPATH=/usr/local/cuda-$CUDA_VERSION/include:$CPATH
#ENV LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:$LIBRARY_PATH
#RUN pip install pycuda

# Install basic ML libralies
RUN pip install torch torchvision keras scikit-learn janome

USER root
RUN apt install -y build-essential curl wget vim git bash-completion