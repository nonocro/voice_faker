from gtts import gTTS
import os
from inferrvc import RVC
import torch.serialization
from fairseq.data.dictionary import Dictionary
from inferrvc import load_torchaudio
import soundfile as sf
import os
import boto3
from flask import Flask, jsonify, make_response, request

def generate_audio(text, language='fr'):
    # Créer un fichier audio avec gTTS
    tts = gTTS(text=text, lang=language)
    audio_path = "audios/audio_output.wav"
    tts.save(audio_path)
    print(f"Fichier audio généré : {audio_path}")
    return audio_path

def modif():
    print("modif")
    os.environ['RVC_MODELDIR'] = 'C:\\voicefaker\\model'  # where model.pth files are stored.
    os.environ['RVC_INDEXDIR'] = 'C:\\voicefaker\\model'  # where model.index files are stored.
    os.environ['RVC_OUTPUTFREQ'] = '44100'  # the audio output frequency, default is 44100.
    os.environ['RVC_RETURNBLOCKING'] = 'True'  # If the output audio tensor should block until fully loaded.

    print("modif2")
    mbappe = RVC('C:\\voicefaker\\model\\Kylian-Mbappe.pth', index='C:\\voicefaker\\model\\added_IVF1022_Flat_nprobe_1_Kylian-Mbappe_v2.index')

    print("modif2.3")

    print(mbappe.name)
    print('Paths', mbappe.model_path, mbappe.index_path)

    print("modif2.1")
    torch.serialization.add_safe_globals([Dictionary])

    print("modif3")
    aud, sr = load_torchaudio('audios/audio_output.wav')

    paudio2 = mbappe(aud, 5, output_device='cpu', output_volume=RVC.MATCH_ORIGINAL, index_rate=.9)

    print("modif4")
    sf.write('audios/audio_mbappe.wav', paudio2, 44100)


app = Flask(__name__)

dynamodb_client = boto3.client('dynamodb')

if os.environ.get('IS_OFFLINE'):
    dynamodb_client = boto3.client(
        'dynamodb', region_name='localhost', endpoint_url='http://localhost:8000'
    )
else:
    dynamodb_client = boto3.client(
        'dynamodb', region_name='localhost', endpoint_url='http://localhost:8000'
    )

USERS_TABLE = os.environ['USERS_TABLE']

@app.route('/test')
def test():
    return jsonify({'message': 'Hello from Flask!'})

@app.route('/generate', methods=['POST'])
def generate_text():
    text_to_transform = request.json.get('text')
    if not text_to_transform:
        return jsonify({'error': 'Please provide "text" parameter'}),
    
    generate_audio(text_to_transform)  # Process the text
    modif()  # Transform the audio

    return jsonify({'message': 'Check your audios folder!'})

@app.errorhandler(404)
def resource_not_found(e):
    return make_response(jsonify(error='Not found!'), 404)
