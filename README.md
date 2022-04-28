# Unconstrained Face Sketch Synthesis

This is a PyTorch implementation of [Unconstrained Face Sketch Synthesis via Perception-Adaptive Network and A New Benchmark](https://www.sciencedirect.com/science/article/pii/S0925231222004660?casa_token=5SR0cXEgOMMAAAAA:ReCvofI1rcmBY_GRxzhtn3KgiwZx5sTvll8TP3qrhDqtzyDXyIyV45OxlNP-iHw21SQZXfQ). In this work, we propose a novel Perception-Adaptive Network (PANet) for face sketch synthesis in the wild. Specifically, our PANet is composed of i) a Fully Convolutional Encoder for hierarchical feature extraction, ii) a Face-Adaptive Perceiving Decoder for extracting potential facial regions and handling face variations, and iii) a Component-Adaptive Perceiving Module for facial component aware feature representation learning. Moreover, we introduce the first medium-scale dataset termed WildSketch for unconstrained face sketch synthesis. This dataset contains 800 pairs of highly-aligned face photo-sketch, which is of better quality and larger scale than the popular CUHK dataset and AR dataset. And it is much more challenging, as there are more variations in pose, age, expression, background clutters, and illumination. 

If you use this code and the WildSketch dataset for your research, please cite our work

```
@article{nie2022unconstrained,
  title={Unconstrained Face Sketch Synthesis via Perception-Adaptive Network and A New Benchmark},
  author={Nie, Lin and Liu, Lingbo and Wu, Zhengtao and Kang, Wenxiong},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```

## Requirements
Install dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Preprocessing
 download the WildSketch dataset from [Dropbox](https://www.dropbox.com/s/kjjwobzxkdouqx1/WildSketch.zip?dl=0) / [BaiDuYun (Password:4jbd)](https://pan.baidu.com/s/1xatF4370gl0mmbMjOv-UsQ) and put it into the folder ```dataset/```.


## Model Training
```bash
sh sketch_training.sh
```

## Testing
Edit ```epoch``` in ```sketch_testing.sh``` to select the testing model.
```bash
sh sketch_testing.sh
```

## Evaluation
Edit ```epoch``` in ```evaluation/sketch_evaluation.m``` to elect the evaluation model.

Matlab is requested to compute the Scoot and FSIM metrics.
```bash
cd evaluation/
matlab -r sketch_evaluation
```

The evaluation of FID can be referred to [here](https://github.com/mseitzer/pytorch-fid). Note that those synthesized sketches should be resized to their original resolutions before computing the FID score.

## Acknowledgements

This project is implemented based on the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
We thank [Dr. JunYan Zhu](https://www.cs.cmu.edu/~junyanz/)!
