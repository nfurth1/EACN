#!/usr/bin/env python

import os
from ase.io import read, write
cwd = os.getcwd()

for filename in os.listdir(cwd):
    if filename.endswith(".xyz"):
        print(filename)
        cell = read(filename)
        name = os.path.splitext(filename)[0]
        #cell[1] = name + "\n"
        write('{}.json'.format(name), cell)
