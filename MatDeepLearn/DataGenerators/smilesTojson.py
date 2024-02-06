#/usr/bin/env python
# Description
import csv
import os
import sys
from ase.io import read, write
from urllib.request import urlopen
from urllib.parse import quote

inFolder = 'in'
idFile = inFolder + '/' + 'targets.csv'
outFolder = 'out'
smiFolder = outFolder + '/smi'
xyzFolder = outFolder + '/xyz'
jsonFolder = outFolder + '/json'

def readInputFile(filename):
    with open(filename) as infile:
        reader = csv.reader(infile, delimiter=',')
        moleculeList = list()
        for row in reader:
            moleculeList.append(row[0])
    return moleculeList

# based on https://stackoverflow.com/questions/54930121/converting-molecule-name-to-smiles
def cirConvert(id):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(id) + '/smiles'
        # print(url)
        ans = urlopen(url).read().decode('utf8')
        print(id + ' worked')
        return ans
    except:
        print(id + ' did not work')
        return False

def getSmiles(ids):
    smiles = dict()
    for id in ids:
        val = cirConvert(id)
        if (val):
            smiles[id] = val
            f = open(smiFolder + '/' + id + '.smi', 'w')
            f.write(val)
            f.close()
    return smiles

# def getXYZ(smiles): # version for input dictionary
    # for name, smile in smiles.items():

def getXYZ():
    for filename in os.listdir(smiFolder):
        name = os.path.splitext(filename)[0]
        smiName = smiFolder + '/' + name + '.smi'
        xyzName = xyzFolder + '/' + name + '.xyz'
        command = 'obabel -ismi "' + smiName + '" -oxyz "' + xyzName + '" --gen3d > "' + xyzName + '"' + ' -ff "MMFF94"'
        print(command)
        os.system(command)

def xyzTojson():
    cwd = os.getcwd() + '/' + xyzFolder
    for fn in os.listdir(cwd):
        if fn.endswith(".xyz"):
            print(fn)
            cell = read(cwd + '/' + fn)
            name = os.path.splitext(fn)[0]
            write('{}.json'.format(jsonFolder + '/' + name), cell)

if not os.path.isdir(inFolder):
    os.mkdir(inFolder)
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)
if not os.path.isdir(smiFolder):
    os.mkdir(smiFolder)
if not os.path.isdir(xyzFolder):
    os.mkdir(xyzFolder)
if not os.path.isdir(jsonFolder):
    os.mkdir(jsonFolder)

if not os.path.isfile(idFile):
    sys.exit('targets.csv is missing :(')

# Get values from the given file (targets.csv usually)
identifiers = readInputFile(idFile)

# print(getSmiles(identifiers))

# Convert those values to smiles
getSmiles(identifiers)

# Convert smiles -> xyz files
#getXYZ()