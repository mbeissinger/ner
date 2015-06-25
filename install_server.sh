#!/usr/bin/env bash

# reference http://markus.com/install-theano-on-aws/
echo "Updating"
sudo apt-get update

echo "Installing Theano and its dependencies"
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# uncomment this next block if you want GPU stuff
#sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
#sudo apt-get update
#sudo apt-get install -y cuda
#echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
#sudo reboot

# if you aren't using the gpu, use the below command, otherwise uncomment the one after and use it instead.
echo -e "\n[global]\nfloatX=float32\ndevice=cpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc
# this is for using the gpu
#echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]=FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >> ~/.theanorc


# Now for OpenDeep
echo "Installing OpenDeep (feature/recurrent_v2 branch)"
git clone -b feature/recurrent_v2 https://github.com/vitruvianscience/opendeep.git
cd opendeep
sudo python setup.py develop
cd ..

# Now you need to clone the NER repo for running the test script (source is https://github.com/mbeissinger/ner)