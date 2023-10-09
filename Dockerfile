#FROM gaetanlandreau/pytorch3d:latest
FROM cpark90/pytorch3d:gl-runtime
#RUN conda -V #install --file requirements.txt

COPY . /Lane-extraction
WORKDIR /Lane-extraction

#RUN apt-get update && apt-get install -y vim
#RUN python -m pip install --upgrade pip
RUN pip install torch-scatter
RUN apt-get install -y vim
RUN python -m pip install -r requirements.txt
RUN pip install spconv-cu117

RUN pip install librosa --user




COPY common/pipeline/main.py ./
CMD ["python","main.py"]
