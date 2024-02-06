#/usr/bin/env python
# Description
import string
import csv
import os
import json
from ase.io import read, write
from urllib.request import urlopen
from urllib.parse import quote
import sys

inFolder = 'in'
idFile = inFolder + '/' + 'targets.csv'
outFolder = 'out'
smiFolder = outFolder + '/smi'
xyzFolder = outFolder + '/xyz'
jsonFolder = outFolder + '/json'

#dictionary from molecule to corresponding smile
smileDic = {}

def readInputFile(filename, extraFeatures, extraFeatureNames):
    extraFeatureList = []

    with open(filename) as infile:
        reader = csv.reader(infile, delimiter=',')
        moleculeList = list()
        next(reader)
        for row in reader:
            moleculeList.append(row[0].rstrip())
            if extraFeatures > 0:
                #gather properties for each molecule and add them to the master list
                moleculeFeats = []
                for j in range(2, extraFeatures + 2):
                    moleculeFeats.append(float(row[j]))
                extraFeatureList.append(moleculeFeats)
    return moleculeList, extraFeatureList

# based on https://stackoverflow.com/questions/54930121/converting-molecule-name-to-smiles
def cirConvert(id):
    #tries to get smile from website, if not checks to see if smile was provided
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(id.rstrip(string.digits)) + '/smiles'
        # print(url)
        ans = urlopen(url).read().decode('utf8')
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
        if os.path.isfile(xyzFolder + '/' + name + '.xyz'):
            i = 1
            while os.path.isfile(xyzFolder + '/' + name + str(i) + '.xyz'):
                i += 1
            xyzName = xyzFolder + '/' + name + str(i) + '.xyz'
        else:
            xyzName = xyzFolder + '/' + name + '.xyz'
        command = 'obabel -ismi "' + smiName + '" -oxyz "' + xyzName + '" --gen3d > "' + xyzName + '"'
        print(command)
        os.system(command)
        # command = 'obabel -ismi "' + smiName + '" -oxyz "'  --gen3d --fastest > "''
        # os.system(command)
        # opName = opFolder + '/' + name + '.xyz'
        # command = 'obminimize -ff MMFF94s -ismi "' + smiName + '" -oxyz "' + opName + '" > "' + opName + '"'
        # os.system(command)

def xyzTojson():
    cwd = os.getcwd() + '/' + xyzFolder
    i = 0
    for fn in os.listdir(cwd):
        if fn.endswith(".xyz"):
            #bug handler
            if os.path.getsize(cwd + '/' + fn) == 0:
                continue
            cell = read(cwd + '/' + fn)
            name = os.path.splitext(fn)[0]
            write('{}.json'.format(jsonFolder + '/' + name), cell)
            if extraFeatures > 0:
                with open('{}.json'.format(jsonFolder + '/' + name), "r+") as file:
                    data = json.load(file)
                    for j in range(extraFeatures):
                        print(extraFeatureList[i][j])
                        data.update({extraFeatureNames[j]: extraFeatureList[i][j]})
                    file.seek(0)
                    json.dump(data, file)
            i += 1

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

extraFeatures = 1
extraFeatureNames = ['Test']

# Get values from the given file (targets.csv usually)
identifiers, extraFeatureList = readInputFile(idFile, extraFeatures, extraFeatureNames)

# Convert smiles -> xyz files
#getSmiles(identifiers)

# getXYZ(smiles)
#getXYZ()

# Convert xyz -> json in out folder
xyzTojson()