# H2SR

## **Environments**

- Pytorch 1.9.1+cu111
- CUDA 11.2
- Python 3.7.1

## Test

```python
python test.py -opt options/test/test.yml
```

- Be sure to download the pre-trained model at [Google](https://drive.google.com/drive/folders/1627IdV9W6pdYNpe4SEmtAConFGGzq1W0?usp=drive_link) before testing.

## Train

The code is built on [SROOE](https://github.com/seungho-snu/SROOE). We thank the authors for sharing the codes.

<br />
If you want to modify the HWNet and Loss:

```
./codes/models/modules/CondNet_arch.py --> HWNet
./codes/models/SRGAN_model.py --> Loss
```
