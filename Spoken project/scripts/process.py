import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import webrtcvad

def float_to_int16(audio_float):
    """
    Convert a float32/float64 NumPy array in the range [-1, 1]
    to 16-bit PCM samples.
    """
    # Ensure the audio is in the range [-1, 1]
    audio_float = np.clip(audio_float, -1.0, 1.0)
    # Scale and convert to int16
    return (audio_float * 32767).astype(np.int16)

def int16_to_float(audio_int16):
    """
    Convert int16 PCM samples to a float NumPy array in the range [-1, 1].
    """
    return audio_int16.astype(np.float32) / 32767.0

def apply_vad(audio_float, sr, frame_duration_ms=30, vad_mode=0):
    """
    Apply WebRTC VAD to the input float audio signal at a given sample rate.
    
    Args:
        audio_float (ndarray): Audio samples in float32/float64, range [-1, 1].
        sr (int): Sampling rate (must be 8000, 16000, or 32000 for WebRTC VAD).
        frame_duration_ms (int): Frame size in milliseconds (10, 20, or 30 are typical).
        vad_mode (int): Aggressiveness mode, 0 = least aggressive, 3 = most aggressive.

    Returns:
        ndarray: Audio signal (float) after removing frames labeled as non-speech.
    """
    # Convert float array to int16 PCM for VAD processing
    audio_int16 = float_to_int16(audio_float)

    # Set up VAD
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)  
    # mode=0 => least aggressive (more likely to label frames as speech)
    # mode=3 => most aggressive (strictest about removing frames)

    # Calculate number of samples per frame
    frame_length = int(sr * frame_duration_ms / 1000)  # e.g., 30 ms * 16000 Hz = 480 samples

    # Break audio into frames
    voiced_frames = []
    start = 0
    while start + frame_length < len(audio_int16):
        frame = audio_int16[start : start + frame_length]
        # WebRTC VAD requires 16-bit PCM bytes
        is_speech = vad.is_speech(frame.tobytes(), sample_rate=sr)
        
        if is_speech:
            voiced_frames.append(frame)
        start += frame_length

    if not voiced_frames:
        # If VAD removed everything, return an empty array (or handle differently)
        return np.array([], dtype=np.float32)

    # Concatenate voiced frames
    voiced_audio_int16 = np.concatenate(voiced_frames, axis=0)
    
    # Convert back to float
    return int16_to_float(voiced_audio_int16)

def preprocess_audio(input_path, output_path, target_sr=16000):
    # 1. Load audio (with original sampling rate)
    audio, sr = librosa.load(input_path, sr=None)
    # print the length of the audio in seconds
    print("file name: ", input_path)
    print(f"Length of audio before VAD: {len(audio)/sr} seconds")
    

    # 2. (Optional) Resample to 16 kHz if needed (necessary for webrtcvad)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 3. Apply VAD at a low aggressiveness level (mode=0)
    audio_vad = apply_vad(audio, sr, frame_duration_ms=30, vad_mode=1)
    if len(audio_vad) == 0:
        # Handle the case where no speech was found
        # For example, just skip writing or write an empty file.
        print(f"No speech detected in {input_path}, skipping.")
        return
    # print the length of the audio after VAD in seconds
    print(f"Length of audio after VAD: {len(audio_vad)/sr} seconds")
    print("*"*50)

    # 4. (Optional) Trim leading/trailing silence if you still want to remove extra quiet parts
    audio_trimmed, _ = librosa.effects.trim(audio_vad, top_db=20)

    # 5. Normalize audio (peak normalization)
    if max(abs(audio_trimmed)) > 0:
        audio_trimmed = audio_trimmed / max(abs(audio_trimmed))

    # 6. Save processed audio
    sf.write(output_path, audio_trimmed, sr)

if __name__ == "__main__":

    # Example: reading from train_metadata.csv
    # with columns ['filename', 'ethnicity'] or similar
    train_df = pd.read_csv('test_metadata.csv')

    for idx, row in train_df.iterrows():
        in_path = os.path.join(r'..\data', 'raw','test', row['filename'])
        out_path = os.path.join(r'..\data', 'processed','test', row['filename'])
        
        preprocess_audio(in_path, out_path)
