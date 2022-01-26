import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch


# Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio Classification
AUDIO_CLASSIFIER_MODEL_NAME = "audio-classifier.pt"
AUDIO_MODEL_PATH = os.path.join(os.getcwd(), f"models/static/{AUDIO_CLASSIFIER_MODEL_NAME}")

# Text Classification

TEXT_CLASSIFIER_MODEL_NAME = "text-emotional-model.pt"
TEXT_MODEL_PATH = os.path.join(os.getcwd(), f"models/static/{TEXT_CLASSIFIER_MODEL_NAME}")

with open(os.path.join(os.getcwd(), f"models/static/text_vocab.json"), 'r') as ref:
    VOCAB = json.load(ref)





