#!bin/bash

for ((size=4 ; size <= 65536 ; size*=2)); do
  echo ./test_lu $size
  ./test_lu $size
done