#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:02:32 2020

@author: gerard
"""


import tkinter as tk
from PIL import ImageTk

root = tk.Tk()

expressions = ['happy', 'surprised', 'sad']
global ind
ind = 0
exp = expressions[ind]

im = ImageTk.PhotoImage(file=f'./faces/{exp}.png')
panel = tk.Label(root, image=im)
panel.pack(fill=tk.BOTH, expand=True)

def callback(e):
    global ind
    ind = (ind + 1) if (ind + 1) < len(expressions) else 0
    exp = expressions[ind]
    img2 = ImageTk.PhotoImage(file=f'./faces/{exp}.png')
    panel.configure(image=img2)
    panel.image = img2

root.bind("<Return>", callback)
root.mainloop()