#!/bin/sh

SCRIPTS_LOCATION=~/machine-learning/tensorflow

docker run -it -v $SCRIPTS_LOCATION:/root b.gcr.io/tensorflow/tensorflow python $1 > output_$1.txt
