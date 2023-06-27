#!/bin/bash

cd schemes && bash gen_img.sh && cd ..

git add .
git commit -m "full update"
git push
