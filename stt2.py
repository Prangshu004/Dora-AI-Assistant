import os
import warnings
import pyaudio
import wave
import audioop
from groq import Groq
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path):
    # ==============================================
    # Audio Recording Configuration
    # ==============================================
    FORMAT = pyaudio.paInt16    #Each audio sample is a 16-bit integer (standard format).
    CHANNELS = 1                #Mono audio (1 channel).
    RATE = 44100                #Sampling rate of 44.1 kHz (CD quality).
    CHUNK = 2048                #Number of audio frames per buffer | Each read from the mic grabs 1024 audio frames at a time.
    SILENCE_THRESHOLD = 6000   # RMS volume level below this = silence. (Lower → more sensitive) 100 default	
    SILENCE_DURATION = 2       # stop after 3 seconds of silence
    #file_path = "auto_stop.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=0,  # Camera index
                        frames_per_buffer=CHUNK)

    logging.info("Recording started... Speak something!")

    frames = []
    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * RATE / CHUNK)

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        rms = audioop.rms(data, 2)  # measure volume
        if rms < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0  # reset if speaking again

        if silent_chunks > max_silent_chunks:
            logging.info("Silence detected — stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

 
    # Save Recorded File
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    logging.info(f"Saved as {file_path}")




# Transcription (Groq Whisper)

def transcribe_with_groq(file_path):
    client=Groq()
    stt_model="whisper-large-v3"
    audio_file=open(file_path, "rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text


# audio_filepath = "test_speech_to_text.mp3"
# record_audio(audio_filepath)
# print(transcribe_with_groq(audio_filepath))