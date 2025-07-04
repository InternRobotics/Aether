# Aether Evaluation

Our evaluation setups mainly follow [CUT3R](https://github.com/CUT3R/CUT3R). Please first follow their instructions to prepare evaluation datasets.

## Video Depth

Make sure you have `sintel`, `kitti`, and `bonn` folders under `data/`.

```console
bash evaluation/video_depth/run_aether.sh
```

The outputs and results can be then found under `eval_results/`.