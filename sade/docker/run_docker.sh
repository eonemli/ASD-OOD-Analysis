#!/bin/bash

NAME="sade-eo"
PORT=9990
# This is the directory where `sade` is cloned
#CODESPACE=/ASD/ahsan_projects/Developer/
CODESPACE=/ASD2/emre_projects/OOD/scripts/sade/

docker stop $NAME-docker || true # Exits gracefully if container doesnt exist

docker run \
	-d \
	--rm \
	--init \
	--name $NAME-docker \
	--ipc=host \
	-e JUPYTER_ENABLE_LAB=yes \
	#--gpus device="3" \
	--gpus '"device=3"' \
	--mount type=bind,src=/ASD2/emre_projects/OOD/braintypicality2/braintypicality/workdir/,target=/workdir/ \
	--mount type=bind,src=/BEE/Connectome/ABCD/,target=/DATA \
	--mount type=bind,src=/ASD2/emre_projects/OOD/scripts/braintypicality-scripts/,target=/braintypicality-scripts/ \
	--mount type=bind,src="/ASD/",target=/ASD \
	--mount type=bind,src="/ASD2/",target=/ASD2 \
	--mount type=bind,src="/UTexas",target=/UTexas \
	--mount type=bind,src=$CODESPACE,target=/codespace \
	-p $PORT:8888 \
	-p 6006:6006 \
	$USER/pytorch_sde:latest \
	jupyter lab --ip 0.0.0.0 --notebook-dir=/ --no-browser --allow-root

	#--mount type=bind,src=/ASD/ahsan_projects/braintypicality/workdir/,target=/workdir/ \