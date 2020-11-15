from __future__ import absolute_import
from __future__ import print_function
import code
import os
import sys
import numpy as np
sys.path.append("/media/ben/datadrive/Software/VGG-Speaker-Recognition/src")
sys.path.append("/media/ben/datadrive/Software/VGG-Speaker-Recognition/tool")
import toolkits
import utils as ut
import pdb
import argparse
import model

from tqdm import tqdm
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score


def get_optimal_balanced_threshold(df):
    rows = []
    for i in np.arange(0, 1, 0.001):
        result = (df["similarity"] > i).value_counts()
        if len(result) == 1:
            # If only True or False then its a bad threshold and move on
            continue
        ratio = float(result.loc[True]) / float(result.loc[False])
        minimise_metric = abs((ratio) -1)
        rows.append({"metric": minimise_metric, "threshold":i, "ratio": ratio})
    _ = pd.DataFrame(rows)
    return _.sort_values("metric").iloc[0]

#  EER: 0.0308370044053
# 2020-11-15_resnet34s_bs16_adam_lr0.001_vlad8_ghost2_bdim512_ohemlevel0/weights-42-0.931.h5

class InferenceEngine:
    def __init__(self):
        self.filename2embedding = {}

        arguments = "--net resnet34s --gpu 0 --ghost_cluster 2 --vlad_cluster 8 --loss softmax " \
                    "--resume " \
                    "/media/ben/datadrive/Software/VGG-Speaker-Recognition/model/gvlad_softmax" \
                    "/2020-11-15_resnet34s_bs16_adam_lr0.001_vlad8_ghost2_bdim512_ohemlevel0" \
                    "/weights-42-0.931.h5 --data_path " \
                    "/media/ben/datadrive/Zalo/voice-verification/Train-Test-Data/dataset/".split()

        ZALO_TEST = "/media/ben/datadrive/Zalo/voice-verification/vgg_db_files/val_trials.txt"

        parser = argparse.ArgumentParser()
        # set up training configuration.
        parser.add_argument("--gpu", default="", type=str)
        parser.add_argument("--resume", default="", type=str)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--data_path", default="/media/weidi/2TB-2/datasets/voxceleb1/wav",
                            type=str)
        # set up network configuration.
        parser.add_argument("--net", default="resnet34s", choices=["resnet34s", "resnet34l"],
                            type=str)
        parser.add_argument("--ghost_cluster", default=2, type=int)
        parser.add_argument("--vlad_cluster", default=8, type=int)
        parser.add_argument("--bottleneck_dim", default=512, type=int)
        parser.add_argument("--aggregation_mode", default="gvlad", choices=["avg", "vlad", "gvlad"],
                            type=str)
        # set up learning rate, training loss and optimizer.
        parser.add_argument("--loss", default="softmax", choices=["softmax", "amsoftmax"], type=str)
        parser.add_argument("--test_type", default="normal", choices=["normal", "hard", "extend"],
                            type=str)
        global args
        args = parser.parse_args(arguments)


        # gpu configuration
        toolkits.initialize_GPU(args)

        # ==================================
        #       Get Train/Val.
        # ==================================
        print("==> Initialising inference engine...".format(args.test_type))

       # ==================================
        #       Get Model
        # ==================================
        # construct the data generator.
        self.params = {"dim": (257, None, 1),
                  "nfft": 512,
                  "spec_len": 250,
                  "win_length": 400,
                  "hop_length": 160,
                  "n_classes": 5994,
                  "sampling_rate": 16000,
                  "normalize": True,
                  }

        self.network_eval = model.vggvox_resnet2d_icassp(input_dim=self.params["dim"],
                                                    num_class=self.params["n_classes"],
                                                    mode="eval", args=args)

        # ==> load pre-trained model ???
        if args.resume:
            # ==> get real_model from arguments input,
            # load the model if the image_model == real_model.
            if os.path.isfile(args.resume):
                self.network_eval.load_weights(os.path.join(args.resume), by_name=True)
                print("==> successfully loading model {}.".format(args.resume))
            else:
                raise IOError("==> no checkpoint found at '{}'".format(args.resume))
        else:
            raise IOError("==> please type in the model to load")

        print("==> start testing.")

    def predict_label(self, audio1_filepath, audio2_filepath):
        if audio1_filepath in self.filename2embedding:
            audio1_embedding = self.filename2embedding[audio1_filepath]
        else:
            audio1_feat = ut.load_data(audio1_filepath,
                                       win_length=self.params["win_length"],
                                       sr=self.params["sampling_rate"],
                                       hop_length=self.params["hop_length"],
                                       n_fft=self.params["nfft"],
                                       spec_len=self.params["spec_len"], mode="eval")
            audio1_feat = np.expand_dims(np.expand_dims(audio1_feat, 0), -1)
            audio1_embedding = self.network_eval.predict(audio1_feat)
            self.filename2embedding[audio1_filepath] = audio1_embedding
        if audio2_filepath in self.filename2embedding:
            audio2_embedding = self.filename2embedding[audio2_filepath]
        else:
            audio2_feat = ut.load_data(audio2_filepath,
                                       win_length=self.params["win_length"],
                                       sr=self.params["sampling_rate"],
                                       hop_length=self.params["hop_length"],
                                       n_fft=self.params["nfft"],
                                       spec_len=self.params["spec_len"], mode="eval")
            audio2_feat = np.expand_dims(np.expand_dims(audio2_feat, 0), -1)
            audio2_embedding = self.network_eval.predict(audio2_feat)
            self.filename2embedding[audio2_filepath] = audio2_embedding

        similarity = np.sum(audio1_embedding * audio2_embedding)

        if similarity > 0.5:
            label = 1
        else:
            label = 0

        return label, similarity


if __name__ == "__main__":
    def validate(df_val):
        df_val = pd.read_csv("../notebooks/df_val.csv")
        with tqdm(total=len(df_val)) as pbar:
            for i, row in df_val.iterrows():
                label, similarity = inference_engine.predict_label(row["path"], row["audio2_path"])
                df_val.loc[i, "similarity"] = similarity
                pbar.update()

        with open(expt_name + "embedding_public.pickle", "wb") as handle:
            pickle.dump(inference_engine.filename2embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df_val.to_csv(expt_name + "_df_val.csv", index=False)

        result = get_optimal_balanced_threshold(df_val)
        print(result)
        df_val["label"] = (df_val["similarity"] > result["threshold"]).astype(int)
        print("Accuracy", accuracy_score(df_val["y_comparison"], df_val["label"]))

    def submit(df_test_sub):
        with tqdm(total=len(df_test_sub)) as pbar:
            for i, row in df_test_sub.iterrows():
                audio1_filepath = os.path.join(expt_root, "Train-Test-Data/public-test/") + str(
                    row["audio_1"])
                audio2_filepath = os.path.join(expt_root, "Train-Test-Data/public-test/") + str(
                    row["audio_2"])
                label, similarity = inference_engine.predict_label(audio1_filepath, audio2_filepath)
                df_test_sub.loc[i, "similarity"] = similarity
                pbar.update()

        with open(expt_name + "embedding_public.pickle", "wb") as handle:
            pickle.dump(inference_engine.filename2embedding, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        df_test_sub.to_csv(expt_name + "_df_all_data.csv", index=False)

        result = get_optimal_balanced_threshold(df_test_sub)
        print(result)
        df_test_sub["label"] = (df_test_sub["similarity"] > result["threshold"]).astype(int)

        df_write = df_test_sub[["audio_1", "audio_2", "label"]]
        df_write.to_csv(expt_name + "_threshold" + str(result['threshold']) + "_submission.csv",
                        index=False)

    expt_name = "42_finetune"
    inference_engine = InferenceEngine()
    expt_root = "/media/ben/datadrive/Zalo/voice-verification/"
    dataset_path = os.path.abspath(os.path.join(expt_root, "Train-Test-Data/public-test.csv"))
    df_test_sub = pd.read_csv(dataset_path)
    submit(df_test_sub)
