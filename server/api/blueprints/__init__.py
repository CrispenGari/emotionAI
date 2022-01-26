from flask import Blueprint, make_response, jsonify, request
from models.oral import classify_audio, audio_classifier_model
from models.textual import predict_text_emotion, text_emotions_model
import io
import soundfile as sf
import torch
import numpy as np

class CustomException(Exception):
    pass

blueprint = Blueprint("blueprint", __name__)
@blueprint.route('/classify/text', methods=["POST"])  
def text_classifier():
    data = {"success": False}
    try:
        if request.method == "POST":
            res = request.get_json(force=True)
            preds = predict_text_emotion(text_emotions_model, res.get("text"))
            data["success"] = True
            data["predictions"] = preds 
            return make_response(jsonify(preds)), 200
        else:
            raise CustomException("the request method should be post only.")
    except Exception as e:
        print(e)
        return make_response(jsonify({
           "message": "internal server error.",
            "code": 500
        })), 500


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