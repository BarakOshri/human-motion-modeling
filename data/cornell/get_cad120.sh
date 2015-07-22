#!/bin/bash
# This script download, untar and preprocess the cornell CAD-120 dataset:
# Link to the dataset: http://pr.cs.cornell.edu/humanactivities/data.php

arr=(   "Subject1_annotations" 
        "Subject3_annotations" 
        "Subject4_annotations" 
        "Subject5_annotations") 

#------------------------------------------------------------------------------#
echo "Downloading..."

for i in "${arr[@]}" 
do 
    if ! [ -d $i ] 
    then
        wget -N --no-check-certificate \
            http://pr.cs.cornell.edu/humanactivities/data/$i.tar.gz
    fi
done

#------------------------------------------------------------------------------#
echo "Untarring..."

for i in "${arr[@]}" 
do 
    if ! [ -d $i ] 
    then
        tar -xzvf $i.tar.gz
    fi

    if [ -a $i.tar.gz ] 
    then
        rm $i.tar.gz
    fi
done

#------------------------------------------------------------------------------#
echo "Prepracessing..."

#------------------------------------------------------------------------------#
echo "Done."


