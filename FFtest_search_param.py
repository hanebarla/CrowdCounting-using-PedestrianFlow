#!/bin/bash

cfitems=(
  "0"
  "0.1"
  "0.01"
  "1.0"
)
setitems=(
  "activate-relu"
  "no_change"
  "opt-amsgrad"
)

echo "===A==="

echo "==== No FF====" >> A_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> A_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu >> A_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar >> A_FFdemo.log
    fi
  done
done

echo "==== Dynamic FF====" >> A_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> A_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 >> A_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 >> A_FFdemo.log
    fi
  done
done

echo "==== Static FF====" >> A_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> A_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --StaticFF 1 >> A_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --StaticFF 1 >> A_FFdemo.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> A_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> A_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> A_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM05.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_A/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 --StaticFF 1 >> A_FFdemo.log
    fi
  done
done



echo "===B==="

echo "==== No FF====" >> B_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> B_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu >> B_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar >> B_FFdemo.log
    fi
  done
done

echo "==== Dynamic FF====" >> B_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> B_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 >> B_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 >> B_FFdemo.log
    fi
  done
done

echo "==== Static FF====" >> B_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> B_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --StaticFF 1 >> B_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --StaticFF 1 >> B_FFdemo.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> B_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> B_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> B_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM01.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_B/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 --StaticFF 1 >> B_FFdemo.log
    fi
  done
done


echo "===C==="

echo "==== No FF====" >> C_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> C_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu >> C_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar >> C_FFdemo.log
    fi
  done
done

echo "==== Dynamic FF====" >> C_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> C_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 >> C_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 >> C_FFdemo.log
    fi
  done
done

echo "==== Static FF====" >> C_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> C_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --StaticFF 1 >> C_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --StaticFF 1 >> C_FFdemo.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> C_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> C_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> C_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM02.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_C/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 --StaticFF 1 >> C_FFdemo.log
    fi
  done
done


echo "===D==="

echo "==== No FF====" >> D_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> D_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu >> D_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar >> D_FFdemo.log
    fi
  done
done

echo "==== Dynamic FF====" >> D_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> D_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 >> D_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 >> D_FFdemo.log
    fi
  done
done

echo "==== Static FF====" >> D_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> D_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --StaticFF 1 >> D_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --StaticFF 1 >> D_FFdemo.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> D_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> D_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> D_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM03.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_D/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 --StaticFF 1 >> D_FFdemo.log
    fi
  done
done


echo "===E==="

echo "==== No FF====" >> E_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> E_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu >> E_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar >> E_FFdemo.log
    fi
  done
done

echo "==== Dynamic FF====" >> E_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> E_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 >> E_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 >> E_FFdemo.log
    fi
  done
done

echo "==== Static FF====" >> E_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> E_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --StaticFF 1 >> E_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --StaticFF 1 >> E_FFdemo.log
    fi
  done
done

echo "==== Dynamic Static FF====" >> E_FFdemo.log
for item in "${cfitems[@]}"; do
  for i in "${setitems[@]}"; do
    echo "=======${item} ${i}=======" >> E_FFdemo.log
    if [ "${i}" = "activate-relu" ]; then
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --activate relu --DynamicFF 1 --StaticFF 1 >> E_FFdemo.log
    else
      python FFtest_search_param.py data/Scene_IM04.csv --dataset CrowdFlow -nw /groups1/gca50095/aca10350zi/habara_exp/FF_confusion_matrix_E/CrowdFlow/${item}/${i}/model_best.pth.tar --DynamicFF 1 --StaticFF 1 >> E_FFdemo.log
    fi
  done
done
