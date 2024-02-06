#!/usr/bin/env bash
inMolecules=("Aspirin" "3-Methylheptane" "Diethylsulfate" "Diethyl sulfate" "Adamant" "50-78-2")

# internal variables
responsebuffer=response.txt
outDir=out

for molecule in "${inMolecules[@]}"; do
    safeMolecule=$(echo $molecule | sed 's/ /%20/g')
    url="https://cactus.nci.nih.gov/chemical/structure/$safeMolecule/smiles"
    echo "getting $molecule at $url"
    http_response=$(curl -s -o $responsebuffer -w "%{http_code}" $url)
    if [ $http_response != "200" ]; then
        #handle
        echo "Didn't find $molecule"
    else
        echo "Found: $molecule"
        # cat response.txt
        mv $responsebuffer "$outDir/$molecule"
    fi
done
rm $responsebuffer
