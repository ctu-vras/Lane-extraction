FROM gaetanlandreau/pytorch3d:0.7.4
#FROM cpark90/pytorch3d:gl-runtime
ENV TORCH_CUDA_ARCH_LIST="Turing Ampere Ada Hopper"
ARG TORCH_CUDA_ARCH_LIST="Turing Ampere Ada Hopper"
COPY . /Lane-extraction
WORKDIR /Lane-extraction

RUN apt-get update && apt-get install -y vim
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN python -m pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN FORCE_CUDA=1 pip install --force-reinstall 'git+https://github.com/facebookresearch/pytorch3d.git'

COPY common/pipeline/main.py ./
CMD ["python","main.py"]
