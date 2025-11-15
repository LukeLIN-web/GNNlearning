#!/bin/bash

imagename=$1
containername=$2
# docker run -it --name tr  pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime  /bin/bash

docker run -it \
    --name $containername \
    $imagename \
    /bin/bash