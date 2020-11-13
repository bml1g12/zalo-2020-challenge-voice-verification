FROM pytorch/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 curl && apt-get clean &&  rm -rf /var/lib/apt/lists/* 

WORKDIR /home/src/

COPY . /home/src

RUN git clone https://github.com/pyannote/pyannote-audio && cd pyannote-audio && git checkout a04134932dfcd0a247f5a8844b9968ed9396e69e && pip install . && cd .. && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir flask flask-restful



