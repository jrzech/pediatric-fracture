# pediatric-fracture

Two directories in this repo correspond to two different modeling approaches.

(1) cnn: weakly-supervised EfficientNetV2-S. Create environment to run this code in conda (assumes you have already installed Anaconda / Miniconda):

```git clone https://www.github.com/jrzech/pediatric-fracture.git
cd pediatric-fracture/cnn
conda env create -f environment.yml
source activate ped-arm-cnn
python -m ipykernel install --user --name ped-arm-cnn --display-name "Python (ped-arm-cnn)"
```

The code for this approach is adapted from code provided at https://github.com/jrzech/reproduce-chexnet.

(2) object-detection: strongly supervised Faster R-CNN based in Detectron2. Creating the environment to run this code requires Docker and is more involved. 
If you do not already have Docker installed, the below demonstrates how I installed it on my system (adapted from https://docs.docker.com/engine/install/ubuntu/): 

```sudo apt-get update

sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
    
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
 sudo apt-get update
 sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
 sudo docker run hello-world
 
 sudo apt-get update

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)       && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -       && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-docker2

sudo systemctl restart docker
```

With docker installed, clone Detectron2 repo
```
git clone https://github.com/facebookresearch/detectron2.git
```
The default Dockerfile at detectron2/docker/Dockerfile did not build for me. Please replace with the Dockerfile provided in this repo at object-detection/docker-config/Dockerfile. Then make sure you are in the detectron2/docker folder and proceed:

```
sudo docker build --build-arg USER_ID=$UID -t detectron2:v0 .
```

Additional helpful files for docker configuration are in this repo in object-detection/docker-config. Once docker image has built 
you can start instance with ./rundet.sh (please note you may need to customize the shared folder, ports, and other details of this for your machine).
Once in the docker image, run ./dockerfinish.sh to install additional needed libraries. 
./jupyter.sh starts a Jupyter Notebook in the docker image.
To remove the docker image, run ./rmdet.sh after exiting the image.

Please note that within the object-detection folder there is a docker-config
folder. This contains the following files: 
