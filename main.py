from gtts import gTTS
import os

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
    try:
        from inferrvc import RVC
        print("RVC imported successfully")
    except Exception as e:
        print(f"Error importing RVC: {e}")
        raise

    mbappe = RVC('C:\\voicefaker\\model\\Kylian-Mbappe.pth', index='C:\\voicefaker\\model\\added_IVF1022_Flat_nprobe_1_Kylian-Mbappe_v2.index')

    print(mbappe.name)
    print('Paths', mbappe.model_path, mbappe.index_path)

    import torch.serialization
    from fairseq.data.dictionary import Dictionary

    torch.serialization.add_safe_globals([Dictionary])

    print("modif3")
    from inferrvc import load_torchaudio
    aud, sr = load_torchaudio('audios/audio_output.wav')

    paudio2 = mbappe(aud, 5, output_device='cpu', output_volume=RVC.MATCH_ORIGINAL, index_rate=.9)

    print("modif4")
    import soundfile as sf
    sf.write('audios/audio_mbappe.wav', paudio2, 44100)

def main(text):
    # Étape 1 : Générer l'audio de base
    generate_audio(text)
    modif()

if __name__ == "__main__":
    text = "Donnez nous des point s'il vous plait. On veut notre diplome."
    main(text)