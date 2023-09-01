FROM gaetanlandreau/pytorch3d:latest
#
#RUN conda -V #install --file requirements.txt

COPY . /Lane-extraction
WORKDIR /Lane-extraction

#RUN pip3 list
RUN apt-get update && apt-get install -y vim
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN pip install pytorch=2.0.0


COPY common/pipeline/main.py ./

#RUN pip install -r requirements.txt
#RUN python3 -m pip install --upgrade pip setuptools wheel
#RUN python3 -m pip install -r requirements.txt
RUN ls
CMD ["python","main.py"]

#ENTRYPOINT ["top", "-b"]