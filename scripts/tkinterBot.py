#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:02:32 2020

@author: gerard
"""

# Interface
import tkinter as tk
from PIL import ImageTk

# Audio
import speech_recognition as sr
import pyaudio
import wave

# Scripts
from nlp_text import chat_pairs

class ChatBot:
    # =============================================================================
    # 1) Create interface: face + text
    # 2) Create mic and record user speech
    # 3) Get machine to answer user
    # 4) Convert text to audio
    # =============================================================================
    
    def __init__(self):
        self.root = tk.Tk()
        self.expressions = ['happy', 'surprised', 'sad']
        self.path_faces = "../faces/"
        # Image
        im = ImageTk.PhotoImage(file=self.path_faces + "happy.png")
        self.panel = tk.Label(self.root, image=im)
        # Text
        self.text = tk.Text(self.root, height=23, width=40)
        self.text.insert(tk.END, "*Press enter to activate ChatBot*\n")
        self.panel.grid(column=0,row=0)
        self.text.grid(column=1,row=0)
        self.face_num = 0
        # Change face with Enter
        self.root.bind("<Return>", self.start)
        # Mic
        self.r = sr.Recognizer()
        self.mic = sr.Microphone(device_index=0)
        # Bot answer
        self.chat_answer = chat_pairs()
        # Start
        self.root.mainloop()
        
    def change_face(self):
        self.face_num = (self.face_num + 1) if (self.face_num + 1) < len(self.expressions) else 0
        exp = self.expressions[self.face_num]
        img2 = ImageTk.PhotoImage(file=f'../faces/{exp}.png')
        self.panel.configure(image=img2)
        self.panel.image = img2
        
    def start(self, event):
        print("Listening...")
        new_text = None
        while (new_text not in ["bye", "good bye", "quit"]):
            with self.mic as source:
                self.r.adjust_for_ambient_noise(source, duration=2) #duration bcs there's a lot of noise here
                audio = self.r.listen(source)
            new_text = self.r.recognize_google(audio, language = "en-IN")
            self.text.insert(tk.END, f"User: {new_text}\n")
            response = self.chat_answer.respond(new_text)
            self.text.insert(tk.END, f"ChatBot: {response}\n")
            
    def diagnosticate(self):
        pass
        
    
        
bot = ChatBot()

