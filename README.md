### Emotion AI

This is a Deep Leaning API for classifying emotions from human face and human audios.

<img src="/1.jpg" alt="alt" width="100%"/>

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

Consist of two parallel models that are trained with different model architectures to save different task. The one is for audio classification and the other is for facial emotion classfication. Each model is served on a different endpoint but on the same server.

### Audio Classification

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

### Classifying audios

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

1. First you need to get the input from `html`
1. Create a `formData` object
1. make a POST requests

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

### Notebooks

If you want to see how the models were trained you can open the respective notebooks:

1. [Audio Classification]()
