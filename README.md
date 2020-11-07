# zalo-2020-challenge-voice-verification

# Strategy

The public test set is 50% match and 50 non-match, so if the model is produces more Matches than Non-Matches, need to adjust the threshold accordingly to optimise accuracy.  

# Brainstorming

Embedding:
* Maybe extracting embedding from just parts of the clip with volume is better?

Hidden features:

* Could the volume or noise characteristics be used as features to identify same speaker (microphone)?
* Some files include some metadata like track id (! can we use this?)
* Track length (! can we use this?)

Public dataset distribution: (TrainTestSetInspection.ipynb)

* A histogram of the number of pairs for a given track has two distinct peaks, a track is compared between 3 and 64 songs, so although the mean is 30 there is almost none with 30 pairs due to the two peaks 
* contains no duplicates based on md5

Training dataset: (TrainTestSetInspection.ipynb)
* Contained duplicates which I deleted or excluded via the usage of a single layer of nesting on the glob

# Notebooks:

SubmitPublicTest.ipynb # First submission - 1_0.174234 using pyannote AMI model, and a 0.66 cosine dist threshold, based on 80% of data split such that 50% is same speaker and 50% not same speaker.


# Results:


| Expt | Public Test | Personal Train  | Personal Val  | notes | 
| ------------- | ------------- | ------------- | ------------- |  ------------- |
| Null  | 0.50  |   |   | |  
| 1 - AMI pretrained  | 0.74234  |  0.79   | | For this I used only first 2 seconds of each clip, but from then on used mean of the clip. |
| 2a - VoxCeleb pretrained | 0.74234  | 0.8645  |   | Threshold selection on my personal train set |
| 2b - VoxCeleb pretrained | 0.74234  |   | | Threshold selection using public set for balanced prec/recall|
| 3 - VoxCeleb pretrained + finetuned 20epoch on my train set| 0.89136 | 0.9285  | 0.9080 | Threshold selection using my val set (0.7003)|

