FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 curl

WORKDIR /home/src/

COPY . /home/src

RUN git clone https://github.com/pyannote/pyannote-audio && cd pyannote-audio && git checkout a04134932dfcd0a247f5a8844b9968ed9396e69e && pip install . && cd .. && pip install -r requirements.txt

RUN pip install flask

RUN pip install flask-restful


