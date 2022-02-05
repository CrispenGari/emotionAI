import os
import torch 
from torch import nn
import torchaudio
from torch.nn import functional as Functional

from models import AUDIO_MODEL_PATH, device

n_input = n_output = 1

# Model Module
class AudioClassifier(nn.Module):
    def __init__(self,
               n_input=1,
               stride=16, 
               n_channel=32,
               emotion_n_output=8,
               emotion_intensity_output=1,
               gender_output=1
               ):
        
        super(AudioClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.emotion_classifier = nn.Sequential(
            nn.Linear(2 * n_channel, 64),
            nn.Linear(64, emotion_n_output)
        )
        self.emotion_intensity_classification = nn.Sequential(
            nn.Linear(2 * n_channel, 64),
            nn.Linear(64, emotion_intensity_output)
        )
        self.gender = nn.Sequential(
            nn.Linear(2 * n_channel, 64),
            nn.Linear(64, gender_output)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = Functional.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        return (self.emotion_classifier(x),
                self.emotion_intensity_classification(x),
                self.gender(x)
            )
# Model instances


print(" ✅ LOADING AUDIO CLASSIFICATION MODEL!")
audio_classifier_model = AudioClassifier().to(device)
audio_classifier_model.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
print("\n ✅ DONE LOADING AUDIO CLASSIFICATION MODEL!\n")



# Transforms
sample_rate , new_sample_rate = (48000, 12000)
transform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                           new_freq=new_sample_rate)


emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
emotion_intensities = ["normal", "strong"]
genders = ["male", "female"]

def pad_sequence(batch):
    batch = torch.nn.utils.rnn.pad_sequence([batch], batch_first=True, padding_value=0.)
    return batch

def preprocess(waveform):
    waveform = pad_sequence(waveform)
    return transform(waveform)


def classify_audio(model, waveform):
    processed = preprocess(waveform).to(device)
    model.eval()
    with torch.no_grad():
        emotion_preds, emotion_intensity_preds, gender_preds = model(processed)

        emotion_pred = torch.softmax(emotion_preds.squeeze(), dim=0)
        emotion_intensity_pred = torch.sigmoid(emotion_intensity_preds.squeeze())
        gender_pred = torch.sigmoid(gender_preds.squeeze())
        
        emotion_intensity_prob = float(emotion_intensity_pred.item()) if emotion_intensity_pred.item() > .5 else float(1 - emotion_intensity_pred.item())
        gender_pred_prob = float(gender_pred.item()) if gender_pred.item() > .5 else float(1 - gender_pred.item())
        emotion_prob = torch.max(emotion_pred).item()
        
        gender_label = 1 if gender_pred.item() >= 0.5 else 0
        emotion_intensity_label = 1 if emotion_intensity_pred.item() >= 0.5 else 0
        emotion_label = torch.argmax(emotion_pred, dim=0).item()
        

        pred =  {
            "emotion": {
                'label': emotion_label,
                'class': emotions[emotion_label],
                'probability':round(emotion_prob, 2),
            },
            "emotion_intensity": {
                'label': emotion_intensity_label,
                'class': emotion_intensities[emotion_intensity_label],
                'probability':round(emotion_intensity_prob, 2),
            },
            "gender": {
                'label': gender_label,
                'class': genders[gender_label],
                'probability':round(gender_pred_prob, 2),
            },
        }
    return pred