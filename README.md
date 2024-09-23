# pediatric-fracture

This repo contains training code and model weights used in the paper [Artificial intelligence to identify fractures on pediatric and young adult upper extremity radiographs](https://pubmed.ncbi.nlm.nih.gov/37740031/) previously published in Pediatric Radiology.

Two directories in this repo correspond to two different modeling approaches.

# Weakly supervised
(1) cnn: weakly-supervised EfficientNetV2-S. Create environment to run this code in conda (assumes you have already installed Anaconda / Miniconda):

```git clone https://www.github.com/jrzech/pediatric-fracture.git
cd pediatric-fracture/cnn
conda env create -f environment.yml
source activate ped-arm-cnn
python -m ipykernel install --user --name ped-arm-cnn --display-name "Python (ped-arm-cnn)"
```

Trained weakly-supervised model can be downloaded from [here](https://drive.google.com/file/d/1IrKFgroRTsw9kmOM2Llo_bQ211y1lISq/view?usp=sharing).


The code for this approach is adapted from code provided at https://github.com/jrzech/reproduce-chexnet.

# Strongly supervised

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

Trained strongly-supervised model can be downloaded from [here](https://drive.google.com/file/d/1pIz2gu8qqc6AeuLRn1yZ8iiAcqPI_npW/view?usp=sharing).

# Annotation format

For those wishing to repurpose the code to their own data, here is an example of the dataset format of the annotations csv file used for the strongly supervised model:

| image_id       | bbox                 | label          | body part | age | sex | height | width | fold  |
|----------------|----------------------|----------------|-----------|-----|-----|--------|-------|-------|
| exam_1_image_1 | [0, 0, 0, 0]         | image_negative | Elbow     | 6   | F   | 1226   | 808   | train |
| exam_1_image_2 | [386, 571, 498, 697] | fracture       | Elbow     | 6   | F   | 1312   | 876   | train |
| exam_1_image_3 | [315, 303, 436, 476] | effusion       | Elbow     | 6   | F   | 881    | 1022  | train |
| exam_1_image_3 | [161, 298, 278, 481] | effusion       | Elbow     | 6   | F   | 881    | 1022  | train |

Please note:

- Each image must show up at least once in the csv. For those with no positive bounding boxes for fracture, an empty [0,0,0,0] box is provided with label 'image_negative.'
- If an image contains multiple findings, each one is included on a separate line (e.g., exam_1_image_3 contains both anterior and posterior elbow effusions, and so appears in two rows above.
- Pixel height and width are required for each image.

I created bounding box annotations using [Prodigy](https://prodi.gy/docs/computer-vision). A Prodigy database can be created from a folder of images using image.manual with the --remove-base64 flag to ensure that the image data is not stored within the database itself (which becomes unwieldy quickly). The db-out function can be used to export a json lines file, from which bounding boxes can be extracted and repackaged into a csv file in the above format.
