# zalo-2020-challenge-voice-verification

SubmitPublicTest.ipynb # First submission - 1_0.174234 using pyannote AMI model, and a 0.66 cosine dist threshold, based on 80% of data split such that 50% is same speaker and 50% not same speaker.

# Notes

Embedding:
* Maybe extracting embedding from just parts of the clip with volume is better?

Hidden features:

* Some files include some metadata like track id
* Track length
* Could the volume or noise characteristics be used as features to identify same speaker (microphone)?

Public dataset distribution: (TrainTestSetInspection.ipynb)

* A histogram of the number of pairs for a given track has two distinct peaks, a track is compared between 3 and 64 songs, so although the mean is 30 there is almost none with 30 pairs due to the two peaks 
* contains no duplicates based on md5

Training dataset: (TrainTestSetInspection.ipynb)
* Contained duplicates which I deleted or excluded via the usage of a single layer of nesting on the glob
