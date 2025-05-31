#!/bin/bash

docker build ./ --build-arg USER=$USER \
				--build-arg UID=$(id -u) \
				-t $USER/pytorch_sde