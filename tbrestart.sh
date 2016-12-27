#!/bin/bash
if [ -z "$1" ]
  then
    echo "Log dir argument requerd"
    exit
fi
pkill -sigkill tensorboard 
tensorboard --logdir=$1

