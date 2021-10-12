## ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot

This repository is the official PyTorch implementation of ICCV-21 paper [ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot](https://arxiv.org/abs/2108.02385).

## Prerequirements
To install the environment.
```bash 
conda env create -f environment.yml
conda activate ACE
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- #### Data format

The annotation of a dataset is a dict consisting of two field: `annotations` and `num_classes`.
The field `annotations` is a list of dict with
`image_id`, `fpath`, `im_height`, `im_width` and `category_id`.

Here is an example.
```
{
    'annotations': [
                    {
                        'image_id': 1,
                        'fpath': '/data/iNat18/images/train_val2018/Plantae/7477/3b60c9486db1d2ee875f11a669fbde4a.jpg',
                        'im_height': 600,
                        'im_width': 800,
                        'category_id': 7477
                    },
                    ...
                   ]
    'num_classes': 8142
}
```

## Usage
#### Training
```bash
#bash data_parallel_train.sh configuration_file_path GPU_indexes
bash data_parallel_train.sh configs/cifar100_im100.yaml 0,1 
```
#### Testing
```bash
#python valid.py configuration_file_path
python valid.py configs/cifar100_im100.yaml
```
[Trained models](https://drive.google.com/drive/folders/19vxHcw-oDB6t3nuLX_iBwljfKtDucj8y?usp=sharing)

## Acknowledgement
This project is developed based on [Bag of tricks @AAAI-21](https://github.com/zhangyongshun/BagofTricks-LT), thanks for their works!

## Citation
```
@inproceedings{cai2021ace,
  title={ACE: Ally Complementary Experts for Solving Long-Tailed Recognition in One-Shot},
  author={Cai, Jiarui and Wang, Yizhou and Hwang, Jenq-Neng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={112--121},
  year={2021}
}
```
