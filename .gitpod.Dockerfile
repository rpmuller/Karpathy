FROM gitpod/workspace-full

#USER gitpod

# Install Julia
# RUN sudo apt-get update \
#     && sudo apt-get install -y \
#         build-essential \
#         libatomic1 \
#         python \
#         gfortran \
#         perl \
#         wget \
#         m4 \
#         cmake \
#         pkg-config \
#         julia \
#     && sudo rm -rf /var/lib/apt/lists/*

#RUN sudo add-apt-repository universe
#RUN sudo add-apt-repository ppa:staticfloat/juliareleases
# RUN sudo add-apt-repository ppa:staticfloat/julia-deps
# RUN sudo apt update
# RUN sudo apt-get upgrade -y

# RUN sudo apt-get install julia

# RUN sudo rm -rf /var/lib/apt/lists/*

RUN cd ~
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.3-linux-x86_64.tar.gz
RUN tar xf julia-1.10.3-linux-x86_64.tar.gz
RUN sudo ln -s ~/julia-1.10.3/bin/julia /usr/local/bin/julia


# Give control back to Gitpod Layer
#USER root
