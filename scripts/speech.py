#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:57:19 2020

@author: gerard
"""

from gtts import gTTS 
import os

text = "Global warming is the long-term rise in the average temperature of the Earth’s climate system"
language = 'en'
speech = gTTS(text = text, lang = language, slow = False)
speech.save("text.mp3")
os.system("nvlc text.mp3") #sudo snap install vlc
