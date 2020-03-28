#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:13:23 2020

@author: ziga
"""
import os
import subprocess
import sys

start = int(sys.argv[1])
stop = int(sys.argv[2])

for i in range(start,stop):
    print i
    subprocess.call("python korona_final.py {0}".format(i),shell=True)
