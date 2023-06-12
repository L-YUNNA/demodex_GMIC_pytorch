# Development of an artificial intelligence model to predict the density and distribution of Demodex species in facial erythema patients
<img src="https://github.com/L-YUNNA/demodex_GMIC_pytorch/assets/129636660/7d3524a7-625f-4ef3-a8b7-87fb1bd0d539" width="650" height="500"/>


<br>안면 홍반 환자의 악화 인자인 모낭충 (demodex species) 밀도를 분류<br/>
<br>모낭충 밀도 측정값은 전문 검사실 인력이 수행한 압출 검사로 확보<br/>
| class | number of demodex |
|-------|-------------------|
|0|0~7ea per 1cm^2|
|1|>10ea per 1cm^2|

-> **Binary Classification**
<br><br/>

## Preprocessing
- De-Identification : [code]https://github.com/L-YUNNA/De-identification
- Missing Value Imputation : 
<br><br/>

## Dataset
- **INTERNAL**<br>
  **total 1,024** (train:valid:test=8:1:1)<br>
  augmentation for trainset
  ![image](https://github.com/L-YUNNA/demodex_GMIC_pytorch/assets/129636660/81386ca2-bc4d-4933-9fa2-5a79fee294e5)
  -> hflip, vflip, shear, rotation, gaussian noise, white balance, color jitter, raw image

- **EXTERNAL**<br>
  total
<br><br/>

## Training methods
- Stratified 10 fold Cross Validation
- Pre-trained CNN (ImageNet)
<br><br/>

## Reference
- An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization
- paper : https://doi.org/10.1016/j.media.2020.101908
- github : https://github.com/nyukat/GMIC
