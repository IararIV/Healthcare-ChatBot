#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:26:40 2020

@author: gerard
"""

import nltk
from nltk.chat.util import Chat, reflections

set_pairs = [
    [
        r"hi|hey|hello",
        ["Hello human, may I know your name?", "Hey there, may I know your name?",]
    ], 
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you?",]
    ],
    [
        r"What is your name?",
        ["My creator calls me IArar, but you can call me a chatbot",]
    ],
    [
        r"how are you ?",
        ["I am fine, thank you! How are you?",]
    ],
    [
        r"I am fine, thank you",
        ["Great to hear that, how can I help you?",]
    ],
    [
        r"I'm not feeling good|My (.*) hurts|I need (.*)|Diagnose me|(.*) bad|(.*)not good|I (.*) feel (.*)",
        ["Oh no, that's not good. Do you mind answering some questions so I can help you?"]
    ],
    [
        r"yes|sure|no problem",
        ["Okay give me a moment\n...",]
    ],
    [
        r"(.*) thank you so much, that was helpful",
        ["Iam happy to help", "No problem, you're welcome",]
    ],
    [
        r"bye|good bye|quit",
    ["Bye, take care. See you soon","It was nice talking to you. Hope you get better"]
],
]

def chat_pairs():   
    chat = Chat(set_pairs, reflections)
    return chat
    