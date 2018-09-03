### Introduction

This is a project which just move the [EM-Adapt](https://bitbucket.org/deeplab/deeplab-public) to tensorflow. The EM-Adapt is referring to the approach for weakly-supervised semantic segmentation in the paper ["Weakly- and semi- supervised learning of a DCNN for semantic image segmentation"](http://liangchiehchen.com/projects/DeepLab.html). And here, I just use the tensorflow to implement the approach with the help of the published code.

### Citing this repository

If you find this code useful in your research, please consider citing them:

> @inproceedings{papandreou15weak,  
>
> ​    title={Weakly- and Semi- Supervised Leaning of a DCNN for Semantic Image Segmentation},
>
> ​    author={George, Papandreou and Liang-Chieh Chen and Kevin Murphy and Alan L Yuille},
>
> ​    journal={arxiv:1502,02734},
>
> ​    year={2015}
>
> }

### Preparation

for using this code, you have to do something else:

##### 1. Download the data and model

1. for pascal data, please referring to its [official website](http://host.robots.ox.ac.uk/pascal/VOC/). Just download it and extract in the ./ .
2. for the init.model, please referring to [EM-ADAPT](http://liangchiehchen.com/projects/Datasets.html). And download it and extract in the mode/ .

For more details, you can referring to the correspond code files or leave a message in the issue.

### Training

then, you just input the following sentence to train it.

> python deeplab.py <gpu_id>

### Result

the final result on the validation dataset of pascal voc 2012 is 37.98% miou while it is 38.2% in the paper. Note that we use the crf while test the trained model, and you can look through my other project to see how to perform densecrf using python.


### Evaluation
I just release a [project](https://github.com/xtudbxk/semantic-segmentation-metrics) to provide the code for evaluation.
