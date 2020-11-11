0. (I deleted some duplicate files from training set, you can see in the TrainTestSetInspection.ipynb - most I did not delete - just exclude by only globbing up to 1 folder down)

1. git clone this branch of pyaudio https://github.com/pyannote/pyannote-audio/pull/504

I used a custom branch here to fix a bug whereby small clips get 100% omitted. 

2. pip install -e . 
(do not pip install pyannote-audio)
3. pip install requirements.txt from this repo, i.e. all the other pyannote stuff in
4. Follow https://github.com/pyannote/pyannote-audio/tree/master/tutorials/data_preparation#audio-files to download AMI dataset (optional)  and to download  MUSAN dataset (required for dataaugmentation)
Made a data folder, on my machine I have:

```
MYFOLDER/MUSAN/
   background_noise.txt
   music.txt
   noise.txt
   speech.txt
MYFOLDER/musan/
   #the data e.g. music/ etc.
MYFOLDER/ZALODATASET/
   development.lst
   development.rttm
   development.trial
   development.uem
   #etc. for train/test. See pyannote_db_files 
```

5. Copy pyannote_db_files/database.yml to MYFOLDER and then set the path of ZALODATASET to where the actual .wav files are on your machine.

6. Add to .bashrc `export PYANNOTE_DATABASE_CONFIG="/MYFOLDER/database.yml"` so pyannote can always find your data

7. git clone https://github.com/juanmc2005/SpeakerEmbeddingLossComparison to get the pretrained model

8. make a folder anywhere (MYFOLDERANYWHEREABSPATH), put in a config.yml describing the network and training setup (e.g. see submission 7) then an example fine-tuning command would be:
```
pyannote-audio emb train --gpu --pretrained /media/ben/datadrive/Software/SpeakerEmbeddingLossComparison/models/AAM/train/VoxCeleb.SpeakerVerification.VoxCeleb2.train/weights/0560.pt --subset train --parallel=8 MYFOLDERANYWHEREABSPATH ZALODATASET.SpeakerVerification.BenProtocol
```
This is saying to train from weight 560.pt using the training subset (i.e. those defined in train.lst/rrtm/trial/uem that I generated in pandas)  and the `emb` and `SpeakerVerification` tell pyannote what sort of training protocol to use.

To evaluate, at the same time, you can run e.g.:

```
pyannote-audio emb validate --gpu --subset development /media/ben/datadrive/Software/pyannote-audio/data/ami/voxceleb_juan_duration2/train/ZALODATASET.SpeakerVerification.BenProtocol.train/ ZALODATASET.SpeakerVerification.BenProtocol
```

*Be aware the specific folder you point to is very important, as it has logic like "go up 2 folders from the provided folder, and find the corresponding config.yml". For training you provide it with the ROOT and for validation (or loading a model) you provide it with the subfolder two below `train`, like the above.*

