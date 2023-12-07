#!bin/bash

for ((size=4 ; size <= 65536 ; size*=2)); do
  echo ./cusolver_lu $size
  ./cusolver_lu $size
done