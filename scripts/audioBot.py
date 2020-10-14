#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:18:06 2020

@author: gerard

https://realpython.com/python-speech-recognition/
"""


import speech_recognition as sr
import pyaudio
import wave

def recorder(WAVE_OUTPUT_FILENAME = "output.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#%%
# =============================================================================
# Getting text from audios
# =============================================================================

r = sr.Recognizer()

#1. Basic
harvard = sr.AudioFile("output.wav")
with harvard as source:
    audio = r.record(source)
    
#2. Duration
harvard = sr.AudioFile("output.wav")
with harvard as source:
    audio = r.record(source, duration=4) #refering to seconds
    
#3. Offset (starting later)
harvard = sr.AudioFile("output.wav")
with harvard as source:
    audio = r.record(source, offset=3) #refering to seconds
    
# Deal with ambient noise
harvard = sr.AudioFile("output.wav")
with harvard as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    audio = r.record(source, offset=3) #refering to seconds

# Get a JSON string with all the alternatives
print(r.recognize_google(audio, show_all='True'))

#%%
# =============================================================================
# Using microphones
# =============================================================================

r = sr.Recognizer()
mic = sr.Microphone(device_index=14)
#print(sr.Microphone.list_microphone_names()) # device_index=3

with mic as source:
    #r.energy_threshold = 0 #257 by default
    r.adjust_for_ambient_noise(source, duration=2)
    audio = r.listen(source)

print(r.recognize_google(audio)) #language='sp-SP'






