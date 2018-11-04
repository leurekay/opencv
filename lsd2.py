#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:07:04 2018

@author: dirac
"""

import cv2
import numpy as np
import os
from pylsd.lsd import lsd
fullName = 'id3.jpg'
folder, imgName = os.path.split(fullName)
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
cv2.imwrite(os.path.join(folder, 'cv2_' + imgName.split('.')[0] + '.jpg'), src)