#!/bin/bash

cfitems=(
  "cf_0"
  "cf_0.1"
  "cf_0.01"
  "cf_1.0"
)
setitems=(
  "activate-relu"
  "no_change"
  "opt-amsgrad"
)

for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/FFtest.csv data/FFtest.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> FFtest.log
    else
      python com_test.py data/FFtest.csv data/FFtest.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --DynamicFF 1 --StaticFF 1 >> FFtest.log
    fi
  done
done