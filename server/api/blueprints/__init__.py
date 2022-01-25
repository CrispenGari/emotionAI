from flask import Blueprint, make_response, jsonify, request
from models import classify_audio, audio_classifier_model
import io
import soundfile as sf
import torch
import numpy as np

blueprint = Blueprint("blueprint", __name__)


@blueprint.route('/classify/audio', methods=["POST"])    
def audio_classifier():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("audio"):
            # read the audio in PIL format
            audio = request.files.get("audio").read()
            audio = io.BytesIO(audio)
            waveform, samplerate = sf.read(file=audio, dtype='float32')
            waveform = torch.from_numpy(np.array([waveform]))
            preds = classify_audio(model=audio_classifier_model, waveform=waveform)
            data["success"] = True
            data["predictions"] = preds    
    return make_response(jsonify(data)), 200
    
    
@blueprint.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "audio classification",
        "description": "given an a audio we will classify the emotions in an audio, gender of the speaker and emotion intensity.",
        "language": "python",
        "library": "pytorch",
        "mainLibray": "torchaudio"
    }
    return make_response(jsonify(meta)), 200