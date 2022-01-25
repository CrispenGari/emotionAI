from app import app
from flask import make_response, jsonify

from blueprints import blueprint
app.register_blueprint(blueprint, url_prefix="/api")

class AppConfig:
    PORT = 3001
    DEBUG = False
    
    
    
@app.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "audio classification",
        "description": "given an a audio of an animal the model should classify weather the sound is for a cat or a dog.",
        "language": "python",
        "library": "pytorch",
        "mainLibray": "torchaudio"
    }
    return make_response(jsonify(meta)), 200

if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )