"""
I set the cropping to "loose" in pyannote.

"""


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
import pickle

expt_name = "embeddings_voxceleb_49epochfinetuned"
emb = Pretrained(validate_dir='/media/ben/datadrive/Software/pyannote-audio/data/ami/voxceleb_finetuneexp2_loose/train/ZALODATASET.SpeakerVerification.BenProtocol.train/validate_equal_error_rate/ZALODATASET.SpeakerVerification.BenProtocol.development/',
                 epoch=49,
                 device="cuda")


os.environ["PYANNOTE_DATABASE_CONFIG"] = "/media/ben/datadrive/Software/pyannote-audio/data/ami/"
# speaker embedding model trained on AMI training set
# emb = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')
expt_root = "/media/ben/datadrive/Zalo/voice-verification/"
dataset_path = os.path.abspath(os.path.join(expt_root, "Train-Test-Data/public-test.csv"))
df_test_sub = pd.read_csv(dataset_path)

filename2embedding = {}
#with open(f"{expt_name}embedding_public.pickle", "rb") as input_file:
#    filename2embedding = pickle.load(input_file)
with tqdm(total=len(df_test_sub)) as pbar:
    for i, row in df_test_sub.iterrows():
        audio1_filepath = os.path.join(expt_root, "Train-Test-Data/public-test/") + str(row["audio_1"])
        if audio1_filepath in filename2embedding:
            audio1_embedding = filename2embedding[audio1_filepath]
        else:
            audio1_embedding = np.mean(emb(
                {"uri": row["audio_1"], "audio": audio1_filepath}),
                                       axis=0, keepdims=True)
            filename2embedding[audio1_filepath] = audio1_embedding

        audio2_filepath = os.path.join(expt_root, "Train-Test-Data/public-test/") + str(row["audio_2"])
        if audio2_filepath in filename2embedding:
            audio2_embedding = filename2embedding[audio2_filepath]
        else:
            audio2_embedding = np.mean(emb(
            {"uri": row["audio_2"], "audio": audio2_filepath}),
                                   axis=0, keepdims=True)
            filename2embedding[audio2_filepath] = audio2_embedding

        # X_audio1 = l2_normalize(np.array([audio1_embedding,]))
        # X_audio2 = l2_normalize(np.array([audio2_embedding,]))
        distance = cdist(audio1_embedding, audio2_embedding, metric="cosine")
        #if (i % 1000) == 0:
        #    print(f"Distance is {distance[0][0]} for index {i} ")
        dist = distance[0][0]
        df_test_sub.loc[i, "dist"] = dist
        df_test_sub.loc[i, "audio1_filepath"] = audio1_filepath
        df_test_sub.loc[i, "audio2_filepath"] = audio2_filepath


        pbar.update()

with open(f"{expt_name}embedding_public.pickle", 'wb') as handle:
    pickle.dump(filename2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

#def get_duration(path):
#    duration = librosa.get_duration(filename=path)
#    return duration
#df_test_sub.loc[:, "audio1_duration"] = df_test_sub["audio1_filepath"].apply(get_duration)
#df_test_sub.loc[:, "audio2_duration"] = df_test_sub["audio2_filepath"].apply(get_duration)
#df_test_sub["duration_dif"] = abs(df_test_sub["audio1_duration"] - df_test_sub["audio2_duration"])

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




result = get_optimal_balanced_threshold(df_test_sub)
df_test_sub["label"] = (df_test_sub["dist"] < result["threshold"]).astype(int)


print(result)
df_write = df_test_sub[["audio_1", "audio_2", "label"]]
df_write.to_csv(f"{expt_name}_threshold{result['threshold']}.csv", index=False)
