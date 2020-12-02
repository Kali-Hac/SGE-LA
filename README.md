# Self-Supervised Gait Encoding with Locality-Aware Attention for Person Re-Identification
By Haocong Rao, Siqi Wang, Xiping Hu, Mingkui Tan, Huang Da, Jun Cheng, Bin Hu. In [IJCAI 2020](https://www.ijcai.org/Proceedings/2020/125).

## Introduction
This is the official implementation of the self-supervised gait encoding model presented by "Self-Supervised Gait Encoding with Locality-Aware Attention for Person Re-Identification".
The codes are used to reproduce experimental results of the proposed Attention-basd Gait Encodings (AGEs) in the [paper](https://www.ijcai.org/proceedings/2020/0125.pdf).

## Requirements
- Python 3.5
- Tensorflow 1.10.0 (GPU)

## Datasets
We provide three already preprocessed datasets (BIWI, IAS, KGBD) on <br/>
[Google Cloud](https://drive.google.com/drive/folders/1apjNcFvlUk9kqnqB1khI3k1HX_cNH46p?usp=sharing) &nbsp; &nbsp; &nbsp;
[Baidu Cloud](https://pan.baidu.com/s/1oOvY2pHM7DFQWcwfVwu6Lw) &nbsp; &nbsp; &nbsp; Password: &nbsp; &nbsp; kle5 &nbsp; &nbsp; &nbsp;
[Tencent Cloud](https://share.weiyun.com/5faKfq4) &nbsp; &nbsp; &nbsp; password：&nbsp; &nbsp; ma385h <br/>
<br />
Two already trained models (BIWI, IAS) are saved in this repository, and all three models can be acquired on <br />
[Google Cloud](https://drive.google.com/drive/folders/1I7eSd37ArGJt46ZfUSzXT0ciDvgW9m-K?usp=sharing) &nbsp; &nbsp; &nbsp;
[Baidu Cloud](https://pan.baidu.com/s/1367Gy-Bk9ojOrXveqCcm0Q) &nbsp; &nbsp; &nbsp; Password: &nbsp; &nbsp; r1jp &nbsp; &nbsp; &nbsp;
[Tencent Cloud](https://share.weiyun.com/5EBPkPZ) &nbsp; &nbsp; &nbsp; password：&nbsp; &nbsp; 6xpj8r  <br/> 
Please download the preprocessed datasets ``Datasets/`` and the model files ``Models/`` into the current directory. 
<br/>

The original datasets can be downloaded from: http://robotics.dei.unipd.it/reid/index.php/downloads (BIWI and IAS-Lab) <br/>
https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor (KGBD) 
 
## Usage

To (1) train the self-supervised gait encoding model to obtain AGEs and (2) validate the effectiveness of AGEs for person Re-ID on a specific dataset with a recognition network,  simply run the following command: 

```bash
# --attention: LA (default), BA  --dataset: BIWI, IAS, KGBD  --gpu 0 (default)
python train.py --dataset BIWI
```
Please see ```train.py``` for more details.

To print evaluation results (Rank-1 accuracy/nAUC) of person re-identification (Re-ID) on the testing set, run:

```bash
# --attention: LA (default), BA  --dataset: BIWI, IAS, KGBD  --gpu 0 (default)
python evaluate.py --dataset BIWI
```

Please see ```evaluate.py``` for more details.

## Citation
```bash
@inproceedings{rao2020self,
	title="Self-Supervised Gait Encoding with Locality-Aware Attention for Person Re-Identification",
	author="Haocong {Rao} and Siqi {Wang} and Xiping {Hu} and Mingkui {Tan} and Huang {Da} and Jun {Cheng} and Bin {Hu}",
	booktitle="IJCAI 2020: International Joint Conference on Artificial Intelligence",
	year="2020"
}

```


## License

SGE-LA is released under the MIT License.

