# Utilities for feature extraction.
import numpy as np
import tempfile
import subprocess
import os
import pretty_midi
import librosa

AUDIO_FS = 22050
AUDIO_HOP = 1024
MIDI_FS = 11025
MIDI_HOP = 512
NOTE_START = 36
N_NOTES = 48

def fast_fluidsynth(m, fs):
    '''Faster fluidsynth synthesis using the command-line program instead of pyfluidsynth.'''
    # Write out temp mid file
    with tempfile.NamedTemporaryFile(delete=False) as temp_mid:
        m.write(temp_mid.name)
    # Get path to temporary .wav file
    with tempfile.NamedTemporaryFile(delete=False) as temp_wav:
        # Get path to default pretty_midi SF2
        sf2_path = os.path.join(os.path.dirname(pretty_midi.__file__), pretty_midi.DEFAULT_SF2)
        # Make system call to fluidsynth
        with open(os.devnull, 'w') as devnull:
            subprocess.check_output(['fluidsynth', '-F', temp_wav.name, '-r', str(fs), sf2_path, temp_mid.name], stderr=devnull)
        # Load from temp wav file
        audio, _ = librosa.load(temp_wav.name, sr=fs)
    # Occasionally, fluidsynth pads a lot of silence on the end, so here we
    # crop to the length of the midi object
    audio = audio[:int(m.get_end_time() * fs)]
    # Remove temp files
    os.remove(temp_mid.name)
    os.remove(temp_wav.name)
    return audio

def midi_cqt(midi_object):
    '''Synthesize MIDI data, compute its constant-Q spectrogram, normalize, and log-scale it.'''
    # Synthesize MIDI object as audio data
    midi_audio = fast_fluidsynth(midi_object, MIDI_FS)
    # Compute CQT of the synthesized audio data
    midi_gram = librosa.cqt(midi_audio, sr=MIDI_FS, hop_length=MIDI_HOP, fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
    # L2-normalize and log-magnitute it
    return post_process_cqt(midi_gram)

def audio_cqt(audio_data, fs=AUDIO_FS):
    '''Compute some audio data's constant-Q spectrogram, normalize, and log-scale it.'''
    # Compute CQT of the synthesized audio data
    audio_gram = librosa.cqt(audio_data, sr=fs, hop_length=AUDIO_HOP, fmin=librosa.midi_to_hz(NOTE_START), n_bins=N_NOTES)
    # L2-normalize and log-magnitute it
    return post_process_cqt(audio_gram)

def post_process_cqt(gram):
    '''Normalize and log-scale a Constant-Q spectrogram.'''
    # Compute log amplitude
    gram = librosa.amplitude_to_db(gram, ref=np.max)  # updated from librosa.logamplitude
    # Transpose so that rows are samples
    gram = gram.T
    # and L2 normalize
    gram = librosa.util.normalize(gram, norm=2., axis=1)
    # and convert to float32
    return gram.astype(np.float32)

def frame_times(gram):
    '''Get the times corresponding to the frames in a spectrogram.'''
    return librosa.frames_to_time(np.arange(gram.shape[0]), AUDIO_FS, AUDIO_HOP)

