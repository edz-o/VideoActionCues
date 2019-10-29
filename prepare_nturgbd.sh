# First download and unzip rawframe.tar and skeleton.tar to data/nturgbd
wget https://www.cs.jhu.edu/~yzh/rawframes.tar
wget https://www.cs.jhu.edu/~yzh/skeletons.tar
mv rawframes.tar data/nturgbd
mv skeletons.tar data/nturgbd
cd data/nturgbd
tar -xf rawframes.tar
tar -xf skeletons.tar
cd ../../

# generate lists for training, you will find the generated *.txt files in data/nturgbd/
cd data_tools/nturgbd
bash generate_rawframes_filelist.sh
cd ../../
