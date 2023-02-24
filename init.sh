mkdir feats
mkdir input
mkdir output
mkdir models

cd input
kaggle competitions download -c march-machine-learning-mania-2023

unzip '*.zip'
rm *.zip