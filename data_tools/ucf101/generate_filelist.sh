#! /usr/bin/bash env

cd ../../
PYTHONPATH=. python data_tools/build_file_list.py ucf101 data/ucf101/rawframes/ --level 1 --format rawframes --shuffle
echo "Filelist for rawframes generated."

PYTHONPATH=. python data_tools/build_file_list.py ucf101 data/ucf101/videos/ --level 1 --format videos --shuffle
echo "Filelist for videos generated."

cd data_tools/ucf101/
