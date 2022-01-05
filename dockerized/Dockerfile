FROM continuumio/miniconda3
WORKDIR /usr/app
COPY ./resources ./
ENTRYPOINT ["python"]
CMD ["app.py"]

RUN pip install -U pip setuptools wheel 
RUN pip install -U spacy 
RUN python -m spacy download en_core_web_sm 

RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

RUN conda install -c conda-forge dash -y
RUN conda install -c conda-forge dash-html-components 
RUN conda install -c conda-forge dash-core-components 
RUN  conda install -c anaconda pandas  -y




