#! /usr/bin/bash env

cd ../
#python build_rawframes.py ../data/nturgbd/videos_train/ ../data/nturgbd/rawframes_train/ --level 1 --ext avi
#echo "Raw frames (RGB only) generated for train set"

python build_rawframes.py ../data/nturgbd/videos_val/ ../data/nturgbd/rawframes_val/ --level 1 --ext avi
echo "Raw frames (RGB only) generated for val set"

cd nturgbd/
