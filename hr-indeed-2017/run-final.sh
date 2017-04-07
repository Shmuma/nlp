#!/usr/bin/env bash
for n in time salary edu exp supervise; do
./train_model.py -r runs/simple-200d.json -n fin-$n -t $n > fin-$n.log 2>&1
done
