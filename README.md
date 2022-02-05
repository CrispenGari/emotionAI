### Emotion AI

This is a Deep Leaning API for classifying emotions from text input and audios.

```
😞 😨 😄 😮 😍 😠 😐 🤮
```

<img src="/1.jpg" alt="alt" width="100%"/>

This api will be able to serve different kind of models to perform emotions predictions based on the following user input.

1. texts
2. audios

### Starting the server

To start the server first you need to install all the packages used by running the following command:

```shell
pip install -r requirements.txt
# make sure your current directory is "server"
```

After that you can start the server by running the following commands:

1. change the directory from `server` to `api`:

```shell
cd api
```

2. run the `app.py`

```shell
python app.py
```

The server will start at a default `PORT` of `3001` which you can configure in the `api/app.py` on the `Config` class:

```py
class AppConfig:
    PORT = 3001
    DEBUG = False
```

> If everything went well you will be able to make api request to the server.

### EmotionAI

Consist of two parallel models that are trained with different model architectures to save different task. The AI api will do the following:

1. Given a text be able to predict the emotions in the text
2. Given an audio be able to predict the emotions in the audio

### Oral emotions

The `8` oral emotions that we will be predicting are as follows:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

<img src="/2.png" alt="alt" width="100%"/>

Sending an audio file to the server at `http://127.0.0.1:3001/api/classify/audio` using the `POST` method we will be able to get the data that looks as follows as the `json` response from the server:

```json
{
  "predictions": {
    "emotion": { "class": "sad", "label": 3, "probability": 0.22 },
    "emotion_intensity": { "class": "normal", "label": 0, "probability": 0.85 },
    "gender": { "class": "male", "label": 0, "probability": 1.0 }
  },
  "success": true
}
```

### Classifying emotions in audios

1. Using `cURL`

To classify the audio using `cURL` make sure that you open the command prompt where the audio files are located for example in my case the audios are located in the `audios` folder so i open the command prompt in the audios folder or else i will provide the absolute path when making a `cURL` request for example

```shell
curl -X POST -F audio=@03-01-01-01-01-01-01.wav http://127.0.0.1:3001/api/classify/audio
```

If everything went well we will get the following response from the server:

```json
{
  "predictions": {
    "emotion": { "class": "sad", "label": 3, "probability": 0.22 },
    "emotion_intensity": { "class": "normal", "label": 0, "probability": 0.85 },
    "gender": { "class": "male", "label": 0, "probability": 1.0 }
  },
  "success": true
}
```

2. Using Postman client

To make this request with postman we do it as follows:

- Change the request method to `POST` at `http://127.0.0.1:3001/api/classify/audio`
- Click on form-data
- Select type to be file on the `KEY` attribute
- For the `KEY` type audio and select the audio you want to predict under value
  `Click` send
- If everything went well you will get the following response depending on the audio you have selected:

```json
{
  "predictions": {
    "emotion": { "class": "sad", "label": 3, "probability": 0.22 },
    "emotion_intensity": { "class": "normal", "label": 0, "probability": 0.85 },
    "gender": { "class": "male", "label": 0, "probability": 1.0 }
  },
  "success": true
}
```

3. Using JavaScript `fetch` api.

- First you need to get the input from `html`
- Create a `formData` object
- make a POST requests

```js
const input = document.getElementById("input").files[0];
let formData = new FormData();
formData.append("audio", input);
fetch("http://127.0.0.1:3001/api/classify/audio", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

If everything went well you will be able to get expected response.

```json
{
  "predictions": {
    "emotion": { "class": "sad", "label": 3, "probability": 0.22 },
    "emotion_intensity": { "class": "normal", "label": 0, "probability": 0.85 },
    "gender": { "class": "male", "label": 0, "probability": 1.0 }
  },
  "success": true
}
```

### Textual Emotions

There are `6` different emotions that we can detect in a sentence or a text which are:

- 😞 -> sadness
- 😨 -> fear
- 😄 -> joy
- 😮 -> surprise
- 😍 -> love
- 😠 -> anger

<img src="/4.png" alt="alt" width="100%"/>

Given a sentence to the right endpoint `http://127.0.0.1:3001/api/classify/text` with expected request body which look as follows:

```json
{
  "text": "some text here"
}
```

The endpoint will call the `textual` emotion classifier and be able to detect emotions in the text and yield the response that looks like:

```json
{
  "predictions": {
    "class_label": "sadness",
    "emoji": "😞",
    "label": 1,
    "predictions": [
      {
        "class_label": "joy",
        "emoji": "😄",
        "label": 0,
        "probability": 0.0
      },
      {
        "class_label": "sadness",
        "emoji": "😞",
        "label": 1,
        "probability": 1.0
      },
      {
        "class_label": "anger",
        "emoji": "😠",
        "label": 2,
        "probability": 0.0
      },
      {
        "class_label": "fear",
        "emoji": "😨",
        "label": 3,
        "probability": 0.0
      },
      {
        "class_label": "love",
        "emoji": "😍",
        "label": 4,
        "probability": 0.0
      },
      {
        "class_label": "surprise",
        "emoji": "😮",
        "label": 5,
        "probability": 0.0
      }
    ],
    "probability": 1.0,
    "sentence": "im updating my blog because i feel shitty."
  },
  "success": true
}
```

1. Classifying emotions on text using `cURL`

To classify the emotion in the text using `cURL` we send the `POST` request as follows:

```shell
curl -X POST http://127.0.0.1:3001/api/classify/text -H "Content-Type: application/json" -d "{\"text\":\"i feel like my irritable sensitive combination skin has finally met it s match.\"}"
```

If everything went well we will be able to se the `json` response that looks as follow:

```json
{
  "class_label": "anger",
  "emoji": "\ud83d\ude20",
  "label": 2,
  "predictions": [
    {
      "class_label": "joy",
      "emoji": "\ud83d\ude04",
      "label": 0,
      "probability": 0.0
    },
    {
      "class_label": "sadness",
      "emoji": "\ud83d\ude1e",
      "label": 1,
      "probability": 0.0
    },
    {
      "class_label": "anger",
      "emoji": "\ud83d\ude20",
      "label": 2,
      "probability": 1.0
    },
    {
      "class_label": "fear",
      "emoji": "\ud83d\ude28",
      "label": 3,
      "probability": 0.0
    },
    {
      "class_label": "love",
      "emoji": "\ud83d\ude0d",
      "label": 4,
      "probability": 0.0
    },
    {
      "class_label": "surprise",
      "emoji": "\ud83d\ude2e",
      "label": 5,
      "probability": 0.0
    }
  ],
  "probability": 1.0,
  "sentence": "i feel like my irritable sensitive combination skin has finally met it s match."
}
```

2. Classifying emotions on text using `POSTMAN` client

To classify the emotions on text using postman client we do it as follows:

- Send a `POST` request at `http://127.0.0.1:3001/api/classify/text`
- Under request body we select `json`
- We add the `json` object that looks as follows:

```json
{
  "text": "i feel like my irritable sensitive combination skin has finally met it s match."
}
```

- Click send and you will be able to see the predictions of the following nature:

```json
{
  "class_label": "anger",
  "emoji": "\ud83d\ude20",
  "label": 2,
  "predictions": [
    {
      "class_label": "joy",
      "emoji": "\ud83d\ude04",
      "label": 0,
      "probability": 0.0
    },
    {
      "class_label": "sadness",
      "emoji": "\ud83d\ude1e",
      "label": 1,
      "probability": 0.0
    },
    {
      "class_label": "anger",
      "emoji": "\ud83d\ude20",
      "label": 2,
      "probability": 1.0
    },
    {
      "class_label": "fear",
      "emoji": "\ud83d\ude28",
      "label": 3,
      "probability": 0.0
    },
    {
      "class_label": "love",
      "emoji": "\ud83d\ude0d",
      "label": 4,
      "probability": 0.0
    },
    {
      "class_label": "surprise",
      "emoji": "\ud83d\ude2e",
      "label": 5,
      "probability": 0.0
    }
  ],
  "probability": 1.0,
  "sentence": "i feel like my irritable sensitive combination skin has finally met it s match."
}
```

3. Using the `javascript` `fetch` API

To classify emotions on text using the javascript fetch api, one can run the following script:

```js
fetch("http://127.0.0.1:3001/api/classify/text", {
  method: "POST",
  headers: new Headers({ "content-type": "application/json" }),
  body: JSON.stringify({
    text: "i feel like my irritable sensitive combination skin has finally met it s match.",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

If everything goes well you will be able to get the following response from the server:

```json
{
  "class_label": "anger",
  "emoji": "\ud83d\ude20",
  "label": 2,
  "predictions": [
    {
      "class_label": "joy",
      "emoji": "\ud83d\ude04",
      "label": 0,
      "probability": 0.0
    },
    {
      "class_label": "sadness",
      "emoji": "\ud83d\ude1e",
      "label": 1,
      "probability": 0.0
    },
    {
      "class_label": "anger",
      "emoji": "\ud83d\ude20",
      "label": 2,
      "probability": 1.0
    },
    {
      "class_label": "fear",
      "emoji": "\ud83d\ude28",
      "label": 3,
      "probability": 0.0
    },
    {
      "class_label": "love",
      "emoji": "\ud83d\ude0d",
      "label": 4,
      "probability": 0.0
    },
    {
      "class_label": "surprise",
      "emoji": "\ud83d\ude2e",
      "label": 5,
      "probability": 0.0
    }
  ],
  "probability": 1.0,
  "sentence": "i feel like my irritable sensitive combination skin has finally met it s match."
}
```

> Note that when you are sending the request to the server using the `javascript` `fetch` API you don't need to worry about `CORS` this is a public API.

### Notebooks

If you want to see how the models were trained you can open the respective notebooks:

1. [Audio (Oral) Emotion Classification](https://github.com/CrispenGari/emotionAI/blob/main/notebooks/01_RAVDESS_EMOTIONS.ipynb)
2. [Textual Emotion Classification](https://github.com/CrispenGari/emotionAI/blob/main/notebooks/01_Emotions_Sentiment_Analyisis.ipynb)
