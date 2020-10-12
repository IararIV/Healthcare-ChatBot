#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:02:32 2020

@author: gerard
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

shape = np.array([256, 512])
screen = np.zeros(shape)

fig = plt.figure(figsize=np.flip(shape)/80)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
plt.imshow(screen, cmap='gray', aspect='auto')
fig.canvas.toolbar.setVisible(False)
plt.show()

#cv2.imwrite('./test.png', screen)

#face1 = np.zeros([20, 33])
