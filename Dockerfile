FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR ~
ARG SYMENGINE_INSTALL_DIR=/usr/local
ARG NEST_INSTALL_DIR=/usr/local

# 0) install dependencies and clone repository

RUN apt-get update && apt-get install -y \
	cmake \
	git \
	libgmp-dev \
	build-essential\
	ghostscript dvipng\
	texlive texlive-latex-extra\
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Happy-Algorithms-League/e2l-cgp-snn.git \
	&& cd e2l-cgp-snn \
	&& git checkout 02bd825
RUN pip3 install -r e2l-cgp-snn/requirements.txt
RUN pip3 install --upgrade cmake
RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 1) install symengine

RUN git clone https://github.com/symengine/symengine.git \
	&& cd symengine  \
	&& mkdir build && cd build \
	&& cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH="${SYMENGINE_INSTALL_DIR}" .. \
	&& make && make install \
	&& ctest \
	&& cd && rm -Rf symengine

# 2, 3 and 4)  install nest version with custom patch

RUN apt-get update && apt-get install -y \
	cython3 \
	libgsl-dev \
	libltdl-dev \
	libncurses-dev \
	libreadline-dev \
	python3-all-dev \
	python3-numpy \
	python3-scipy \
	python3-matplotlib \
	python3-nose \
	openmpi-bin \
	libopenmpi-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nest/nest-simulator.git \
	&& cd nest-simulator \
	&& git checkout 27dc242 \
	&& git apply ../e2l-cgp-snn/nest/ltl_usrl_us_sympy.patch \
	&& mkdir build && cd build \
	&& cmake -Dwith-python=3 -DCMAKE_INSTALL_PREFIX:PATH="${NEST_INSTALL_DIR}" .. \
	&& make && make install \
	&& cd && rm -Rf nest-simulator
RUN echo "source ${NEST_INSTALL_DIR}/bin/nest_vars.sh" >> ~/.bashrc 

# 5) install custom extensions

ENV NEST_MODULE_PATH=/usr/local/lib/nest:$NEST_MODULE_PATH
ENV SLI_PATH=/usr/local/share/nest/sli:$SLI_PATH

RUN cd e2l-cgp-snn/nest/stdp-homeostatic-synapse-module/ \
	&& mkdir build && cd build \
	&& cmake -Dwith-nest="${NEST_INSTALL_DIR}/bin/nest-config" ../ \
	&& make && make install
RUN cd e2l-cgp-snn/nest/usrl-synapse-module/ \
	&& mkdir build && cd build \
	&& cmake -Dwith-nest="${NEST_INSTALL_DIR}/bin/nest-config" ../ \
	&& make && make install
RUN cd e2l-cgp-snn/nest/us-sympy-synapse-module/ \
	&& mkdir build && cd build \
	&& cmake -Dwith-nest="${NEST_INSTALL_DIR}/bin/nest-config" -Dwith-symengine="${SYMENGINE_INSTALL_DIR}" ../ \
	&& make && make install
RUN cd e2l-cgp-snn/nest/stdp-sympy-synapse-module/ \
	&& mkdir build && cd build \
	&& cmake -Dwith-nest="${NEST_INSTALL_DIR}/bin/nest-config" -Dwith-symengine="${SYMENGINE_INSTALL_DIR}" ../ \
	&& make && make install

ENV PYTHONPATH=/usr/local/lib/python3.6/site-packages/:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/include:$LD_LIBRARY_PATH
ENV NEST_MODULE_PATH="${NEST_INSTALL_DIR}/lib/nest:$NEST_MODULE_PATH"
ENV SLI_PATH="${NEST_INSTALL_DIR}/share/nest/sli:$SLI_PATH"
