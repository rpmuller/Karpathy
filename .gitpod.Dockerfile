FROM gitpod/workspace-full

ENV JULIA_PATH /usr/local/julia
ENV PATH $JULIA_PATH/bin:$PATH

USER gitpod

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.3-linux-x86_64.tar.gz
RUN sudo mkdir $JULIA_PATH
RUN sudo tar -xzf julia-1.10.3-linux-x86_64.tar.gz -C $JULIA_PATH --strip-components 1
RUN julia --version

# Give control back to Gitpod Layer
USER root
