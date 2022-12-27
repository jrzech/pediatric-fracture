sudo docker run -p 9976:9976 -p 6006:6006 --shm-size 20G --gpus all -it -v ~/research/DOCKERSHARED:/home/appuser/detectron2_repo/SHARED --name=detectron2_container detectron2:v0
