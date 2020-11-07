from tqdm import tqdm
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from pyannote.audio.features import Pretrained
from pyannote.core.utils.distance import l2_normalize
from pyannote.core.utils.distance import cdist
import os

threshold = 0.792 # from val set
expt_name = "embeddings_voxceleb_20epochfinetuned"
emb = Pretrained(validate_dir='/media/ben/datadrive/Software/pyannote-audio/data/ami/tmp2/train/ZALODATASET.SpeakerVerification.MixHeadset.train/validate_equal_error_rate/ZALODATASET.SpeakerVerification.MixHeadset.val/',
                 epoch=20)


os.environ["PYANNOTE_DATABASE_CONFIG"] = "/media/ben/datadrive/Software/pyannote-audio/data/ami/"
# speaker embedding model trained on AMI training set
# emb = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')
expt_root = "/media/ben/datadrive/Zalo/voice-verification/"
dataset_path = os.path.abspath(os.path.join(expt_root, "Train-Test-Data/public-test.csv"))
df_test_sub = pd.read_csv(dataset_path)
with tqdm(total=len(df_test_sub)) as pbar:
    for i, row in df_test_sub.iterrows():
        audio1_embedding = np.mean(emb(
            {"uri": row["audio_1"], "audio": os.path.join(expt_root, "Train-Test-Data/public-test/") + str(row["audio_1"])}),
                                   axis=0, keepdims=True)
        audio2_embedding = np.mean(emb(
            {"uri": row["audio_2"], "audio":  os.path.join(expt_root, "Train-Test-Data/public-test/") + str(row["audio_2"])}),
                                   axis=0, keepdims=True)
        # X_audio1 = l2_normalize(np.array([audio1_embedding,]))
        # X_audio2 = l2_normalize(np.array([audio2_embedding,]))
        distance = cdist(audio1_embedding, audio2_embedding, metric="cosine")
        #if (i % 1000) == 0:
        #    print(f"Distance is {distance[0][0]} for index {i} ")
        dist = distance[0][0]
        df_test_sub.loc[i, "dist"] = dist
        if dist < threshold:
            df_test_sub.loc[i, "label"] = 1
        else:
            df_test_sub.loc[i, "label"] = 0

        pbar.update()

df_write = df_test_sub[["audio_1", "audio_2", "label"]]
df_write.to_csv(f"{expt_name}.csv", index=False)


def get_optimal_balanced_threshold(df):
    rows = []
    for i in np.arange(0, 1, 0.001):
        result = (df["dist"] < i).value_counts()
        if len(result) == 1:
            # If only True or False then its a bad threshold and move on
            continue
        ratio = result.loc[True] / result.loc[False]
        minimise_metric = abs((ratio) -1)
        rows.append({"metric": minimise_metric, "threshold":i, "ratio": ratio})
    _ = pd.DataFrame(rows)
    return _.sort_values("metric").iloc[0]
print(get_optimal_balanced_threshold(df_test_sub))
