from flask import Flask, request, jsonify
import json
import os
from werkzeug.utils import secure_filename
from inference import InferenceEngine

AUDIO_STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_storage")
if not os.path.isdir(AUDIO_STORAGE):
    os.makedirs(AUDIO_STORAGE)

# Flask
app = Flask(__name__)

inference_engine = InferenceEngine()

@app.route("/api/predict", methods=['POST'])
def api_predict():
    """
    Example usage
    curl -F 'audio_1=@zzO59XYRJSSIy20G8RQK.wav' -F 'audio_2=@zzO59XYRJSSIy20G8RQK.wav' http://0.0.0.0:6677/api/predict
    >>
    {
      "label": null
    }

    Required params:
        audio_1, audio_2
    """
    print("The request.files contains: ",  request.files)
    audio_file_1 = request.files['audio_1']  # Required
    audio_file_2 = request.files['audio_2']  # Required
    if audio_file_1:
        filename_1 = secure_filename(audio_file_1.filename)
        audio_file_1.save(os.path.join(AUDIO_STORAGE,
                                       filename_1))  # Save audio in audio_storage,
        # path: audio_storage/filename_1

    if audio_file_2:
        filename_2 = secure_filename(audio_file_2.filename)
        audio_file_2.save(os.path.join(AUDIO_STORAGE,
                                       filename_2))  # Save audio in audio_storage,
        # path: audio_storage/filename_2

    # Import code here
    print(f"Performing inference on {os.path.join(AUDIO_STORAGE, filename_1)} and"
          f" {os.path.join(AUDIO_STORAGE, filename_2)}")

    label = inference_engine.predict_label(os.path.join(AUDIO_STORAGE, filename_1),
                                           os.path.join(AUDIO_STORAGE, filename_2))

    print(f"Returning label = {label}")
    return jsonify(
        label=label
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6677', debug=True)
