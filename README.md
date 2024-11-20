# EffIntentGCN

## Introduction
Test code for EffIntentGCN: An Efficient Graph Convolutional Network for Skeleton-based Pedestrian Crossing Intention Prediction

## Data
Please get the skeleton data from [Pedestrian Crossing Action Prediction Benchmark](https://github.com/ykotseruba/PedestrianActionBenchmark).
1. Visit the repository and follow the provided instructions to download the JAAD data.
2. Set the following parameters
   - `obs_length = 32`  
   - `time_to_event = [30, 60]`  
   - `overlap = 1`  
   - `obs_input_type = pose`  
   - `datasets = jaad_beh`  
After setting these parameters, save the obtained skeleton data as `jaad_beh_data.pkl`.
3. Move the generated file to the `./data` directory within project.  

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