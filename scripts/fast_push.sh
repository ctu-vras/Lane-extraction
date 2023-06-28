#!/bin/bash

cd doc && bash gen_img.sh && cd ..

git add .
git commit -m "full update"
git push
