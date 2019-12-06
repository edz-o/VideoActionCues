# Optical Flow

## Data Preparation

To extract optical flow from all RGB frames, run 
```shell
bash extract_flow.sh
```

## Training

To train the network with optical flow, run

```shell
bash train_denseflow.sh
```

## Test Pretrained Model

To test the network on the test set, run

```shell
bash test_denseflow.sh
```
