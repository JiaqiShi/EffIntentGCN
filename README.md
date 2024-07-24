# EffIntentGCN

## Introduction
Test code for EffIntentGCN: An Efficient Graph Convolutional Network for Skeleton-based Pedestrian Crossing Intention Prediction

## Data
Please get the skeleton data from [Pedestrian Crossing Action Prediction Benchmark](https://github.com/ykotseruba/PedestrianActionBenchmark).

## Running

To test on the joint data, please run

```
python test_model.py --ckps_path ckps/skeleton
```

To test on the bone stream, please run

```
python test_model.py --ckps_path ckps/bone
```

To test the two stream model, please run

```
python test_model.py --ckps_path ckps/two-stream
```