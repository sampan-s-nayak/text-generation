from flask import Flask, jsonify, request
from Settings import settings as default
from Settings import dataset
import os
from Model import model as net

app = Flask(__name__)

# load the pretrained model
model = net.load_model(path=os.path.join(default.WEIGHTS_PATH,dataset.WEIGHTS))
model.to(default.DEVICE)

@app.route('/generate_text/', methods=['POST'])
def generate_bootstrap():
    if request.method == 'POST':
        # gather the required data 
        initial_text = request.args.get('initial_text')
        length = int(request.args.get('length'))

        # generate text
        generated = net.generate_text(model,initial_text,length)

    return jsonify({
        'input':initial_text,
        'lenght':length,
        'output':generated,
    })