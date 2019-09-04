#!/bin/bash
if [ -n "$1" ]
then
export EXT_FLAG="$1"
fi
sudo WITH_CUDA=ON $EXT_FLAG python setup.py install
