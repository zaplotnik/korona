#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 01:23:50 2019

@author: ziga
"""

import sys
import os
import subprocess
import time

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

nodes=[14]
j = 0
node = nodes[j]
replace_line("run_job.sh",1,"#SBATCH --nodelist=node{:02d}\n".format(node))

max_pnode = 32

for k in range(0,32):
    node = nodes[k//max_pnode]
    print(k,node)

    replace_line("run_job.sh",1,"#SBATCH --nodelist=node{:02d}\n".format(node))
    
    print k,k+1    
    replace_line("run_job.sh", 17, "python run_korona.py {0} {1}".format(k,k+1))
    
    subprocess.call("sbatch run_job.sh", shell=True)
    time.sleep(1)
