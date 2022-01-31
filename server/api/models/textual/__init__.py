import torch, spacy, os
from torch import nn
from models import VOCAB, device, TEXT_MODEL_PATH
from torch.nn import functional as F
import numpy as np
import emoji

classes = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

emotions_emojis = {
   'anger' : ":angry:", 
   'fear': ":fearful:", 
   'joy' : ":smile:", 
   'love' : ":heart_eyes:", 
   'sadness' : ":disappointed:", 
   'surprise': ":open_mouth:"
}
base_dir = os.path.join(os.getcwd(), "translation/models")
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

class EmotionsLSTMRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size,
               hidden_size, output_size, num_layers,
               bidirectional, dropout, pad_index
               ):
        super(EmotionsLSTMRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_size,
                                    padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_size, hidden_size = hidden_size,
                            bidirectional=bidirectional, num_layers=num_layers,
                            dropout = dropout
                            )
        self.hidden_1 = nn.Linear(hidden_size * 2, out_features=512)
        self.hidden_2 = nn.Linear(512, out_features=256)
        self.output_layer = nn.Linear(256, out_features=output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), enforce_sorted=False)
        packed_output, (h_0, c_0) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        h_0 = self.dropout(torch.cat((h_0[-2,:,:], h_0[-1,:,:]), dim = 1))

        out = self.dropout(self.hidden_1(h_0))
        out = self.hidden_2(out)
        return self.output_layer(out)


# Tokenizers
print(" ✅ LOADING TOKENIZERS\n")
spacy_en = spacy.load('en_core_web_sm')
print(" ✅ LOADING TOKENIZERS DONE!\n")

def tokenize_en(sent: str) -> list:
    return [tok.text for tok in spacy_en.tokenizer(sent)]

INPUT_DIM = 15167

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 6
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = VOCAB.get(PAD_TOKEN) 

print(" ✅ LOADING TEXT EMOTION MODELS\n")
text_emotions_model = EmotionsLSTMRNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX).to(device)

text_emotions_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=device))
print(" ✅ LOADING TRANSLATION MODELS DONE!\n")


def predict_text_emotion(model, sentence:str, min_len = 5):
    model.eval()
    with torch.no_grad():
        tokenized = tokenize_en(sentence.lower())
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [VOCAB.get(t) for t in tokenized]
        indexed = [i if i is not None else VOCAB.get('<unk>') for i in indexed]
        length =  [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        probabilities = F.softmax(model(tensor, length_tensor), dim=1).detach().cpu().numpy().squeeze()
        prediction = np.argmax(probabilities, axis=0)

        all_preds = [
            {
            'label': int(i),
            'class_label': classes[i],
            'probability': float(np.round(probabilities[i], 2)),
            'emoji': emoji.emojize(emotions_emojis[classes[i]], language='en', use_aliases=True)
            } for i, _ in enumerate(probabilities)
        ]

        res ={
            "sentence": sentence,
            'label': int(prediction),
            'class_label': classes[prediction],
            'probability': float(np.round(probabilities[prediction], 2)),
            'predictions': all_preds,
            'emoji': emoji.emojize(emotions_emojis[classes[prediction]], language='en', use_aliases=True)
        }
        return res
