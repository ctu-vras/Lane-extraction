#!/bin/bash

for f in *.drawio; do
    drawio -x -f png --scale 1.5 \
        -o images/${f%.*}.png $f
done

#drawio -x -f png --scale 2.5 \
#    -o published-diagram.png mydiagram.svg
