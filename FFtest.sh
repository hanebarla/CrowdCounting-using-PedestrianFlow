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

echo "===IM04==="

echo "==== No FF====" >> Scene_IM04_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM04_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu >> Scene_IM04_FFtest.log
    else
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar >> Scene_IM04_FFtest.log
    fi
  done
done

echo "==== Dynamic FF====" >> Scene_IM04_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM04_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --DynamicFF 1 >> Scene_IM04_FFtest.log
    else
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --DynamicFF 1 >> Scene_IM04_FFtest.log
    fi
  done
done

echo "==== Static FF====" >> Scene_IM04_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM04_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --StaticFF 1 >> Scene_IM04_FFtest.log
    else
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --StaticFF 1 >> Scene_IM04_FFtest.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> Scene_IM04_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM04_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> Scene_IM04_FFtest.log
    else
      python com_test.py data/Scene_IM04.csv data/Scene_IM04.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --DynamicFF 1 --StaticFF 1 >> Scene_IM04_FFtest.log
    fi
  done
done

echo "===IM05==="

echo "==== No FF====" >> Scene_IM05_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM05_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu >> Scene_IM05_FFtest.log
    else
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar >> Scene_IM05_FFtest.log
    fi
  done
done

echo "==== Dynamic FF====" >> Scene_IM05_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM05_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --DynamicFF 1 >> Scene_IM05_FFtest.log
    else
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --DynamicFF 1 >> Scene_IM05_FFtest.log
    fi
  done
done

echo "==== Static FF====" >> Scene_IM05_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM05_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --StaticFF 1 >> Scene_IM05_FFtest.log
    else
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --StaticFF 1 >> Scene_IM05_FFtest.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> Scene_IM05_FFtest.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> Scene_IM05_FFtest.log
    if [ "${i}" = "activate-relu" ]; then
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> Scene_IM05_FFtest.log
    else
      python com_test.py data/Scene_IM05.csv data/Scene_IM05.csv --dataset CrowdFlow --load_model /home/data/TrainedModel/CrowdFlow/${item}/${i}/checkpoint.pth.tar --DynamicFF 1 --StaticFF 1 >> Scene_IM05_FFtest.log
    fi
  done
done
