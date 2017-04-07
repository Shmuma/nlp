#!/usr/bin/env bash
for n in time salary edu exp; do # supervise
./train_model.py -r runs/simple-200d.json -n fin-$n -t $n > fin-$n.log 2>&1
done
