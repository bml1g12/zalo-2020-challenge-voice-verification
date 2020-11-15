# zalo-2020-challenge-voice-verification

# Strategy

The public test set is 50% match and 50 non-match, so if the model is produces more Matches than Non-Matches, need to adjust the threshold accordingly to optimise accuracy.  

# To do before submission

* fine-tune on entire train set
* Try shorter training with shorter duration for the embedddings, as I have a lot of micro clips https://github.com/pyannote/pyannote-audio/issues/458
* https://github.com/didi/delta try this repo --> seems to report state of the art EER 
* Pyannote --> add augmentation, the paper says TripletLoss performs better
* What are the failure cases? Does it fail on longer clips more than smaller clips? Can a human discern the failure cases? Are there specific background sounds in the failure cases etc.

# Brainstorming

* Maybe extracting embedding from just parts of the clip with volume is better?
* We have gender labels for training, could consider using a pretrained or even fine-tuned gender model as a way of eliminating pairing audio1 with audio2 if different genders.


Hidden features:

* Could the volume or noise characteristics be used as features to identify same speaker (microphone)?
* Public set has no metadata except: streaminfo_duration	filesize	streaminfo__size	frame_count	 byte_count. These are cross correlated  with duration/size, and although there are patterns in the train set, it seems the public set has uniform ratios between them. So no data leakage in public set as far as I can see.
* The training partition I made shows a difference in durations between audio1 and audio2 of > 26.49s is 100% correlated with not being a match, presumably as they come from dif. sources. Public test set expt. 6 suggested this is not useful filter.

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
| 4 - VoxCeleb pretrained + finetuned changed sourcecode to "loose" from here on. | 0.8892 | | | Threshold using set for balanced prec/recall. 49 epochs based on optimum EER on my val set (0.084)|
| 5 - VoxCeleb pretrained (repeat of 4 at epoch 19). | 0.8899 | | | Threshold using set for balanced prec/recall. 19 epochs. EER on my val set 0.091 (0.084). |
| 6 - VoxCeleb pretrained (repeat of 4 at epoch 19). | 0.88948 | | | tried setting all items with large duration disparities between Audio1 and Audio2 (> 26.49s) as non-match. reduced accuracy so I guess this is not a useful feature. |
| 7 - VoxCeleb pretrained using source [0] "juan" |0.90534 | | 0.91515 | Note that I did an experiment locally whereby I used the default config.yml in this repo, and trained for 20 epoch but got same personal0-val as 6. I noticed many clips are small, and so made some changes duration: 0.75 (default 3). I noticed we might not have a huge amount of audio from each speaker so set: label_min_duration: 10 (default 30). I put batch size 512 (no real reason - trying to make it train faster!)|
| 7a | | | 0.9303 (new val set)  | Fixed the bug issue with .trials and df_val having wrong labels; actually not ensurely sure the labels were wrong (just the uris) but repeating 7, I selected the best ERR(epoch 31, 0.0495841 EER on the new val set and submit.|
| 8 | did not bother to subit | 0.931|  (new val set)  | Repeat of 7a but min speaker duration 1. epoch 30 Val EER 0.048865|

__Sources__:
[0] https://github.com/juanmc2005/SpeakerEmbeddingLossComparison 

>>>>>>> 6677292c97284125a4b5b96d425ba1298045791a
# Pyannote tips

Line 125.py of audio/applications/speaker_embedding.py needed editing https://github.com/pyannote/pyannote-audio/issues/471

 pyannote-audio emb train --pretrained=emb_voxceleb --subset=train --parallel=8 /media/ben/datadrive/Software/pyannote-audio/data/ami/voxceleb_finetuneexp1/ ZALODATASET.SpeakerVerification.BenProtocol

 pyannote-audio emb validate  --subset=development --parallel=8 /media/ben/datadrive/Software/pyannote-audio/data/ami/voxceleb_finetuneexp1/train/ZALODATASET.SpeakerVerification.MixHeadset.train ZALODATASET.SpeakerVerification.BenProtocol
