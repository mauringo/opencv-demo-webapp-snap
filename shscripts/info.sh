#!/bin/bash

echo "The Snap configuration files are stored in:"
echo $SNAP_DATA

echo $HOME
echo $USER
env

path="/home/$(ls /home)"
echo $path
exit 0



