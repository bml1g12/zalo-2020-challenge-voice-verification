import torch
import numpy as np
import pandas as pd
from pyannote.audio.features import Pretrained
from pyannote.core.utils.distance import cdist
import os


class InferenceEngine:
    def __init__(self):
        self.engine = Pretrained(
            validate_dir='/home/src/model_files/ZALODATASET.SpeakerVerification.BenProtocol.train/validate_equal_error_rate/ZALODATASET.SpeakerVerification.BenProtocol.development/',
            epoch=31,
            device="cpu")

        self.filename2embedding = {}


    def predict_label(self, audio1_filepath, audio2_filepath):
        if audio1_filepath in self.filename2embedding:
            audio1_embedding = self.filename2embedding[audio1_filepath]
        else:
            audio1_embedding = np.mean(self.engine(
                {"uri": "audio1uri", "audio": audio1_filepath}),
                axis=0, keepdims=True)
            self.filename2embedding[audio1_filepath] = audio1_embedding

        if audio2_filepath in self.filename2embedding:
            audio2_embedding = self.filename2embedding[audio2_filepath]
        else:
            audio2_embedding = np.mean(self.engine(
                {"uri": "audio2uri", "audio": audio2_filepath}),
                axis=0, keepdims=True)
            self.filename2embedding[audio2_filepath] = audio2_embedding

        distance = cdist(audio1_embedding, audio2_embedding, metric="cosine")

        dist = distance[0][0]

        if dist < 0.5:
            label = 1
        else:
            label = 0

        return label
