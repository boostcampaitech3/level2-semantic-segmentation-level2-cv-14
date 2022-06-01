# One Team

![Untitled](https://user-images.githubusercontent.com/64190071/164357692-6ab59eb0-d522-495b-ba49-60221bd4f6a6.png)

- 2022.04.25 ~ 2022.05.17
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- 재활용 품목 분류를 위한 Semantic Segmentation

## MEMBERS

|                                                  [김찬혁](https://github.com/Chanhook)                                                   |                                                                          [김태하](https://github.com/TaehaKim-Kor)                                                                           |                                                 [문태진](https://github.com/moontaijin)                                                  |                                                                        [이인서](https://github.com/Devlee247)                                                                         |                                                                         [장상원](https://github.com/agwmon)                                                                         |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![KakaoTalk_20220421_103825891_03](https://user-images.githubusercontent.com/64190071/164358205-2048f3c2-1216-4836-a77f-a25de6a9091c.jpg) | ![KakaoTalk_20220421_103825891](https://user-images.githubusercontent.com/64190071/164358113-c8db12e4-15d1-469c-8026-cd0a5cb89e36.jpg) | ![KakaoTalk_20220421_103825891_04](https://user-images.githubusercontent.com/64190071/164358227-ef0d7919-bd0d-4a9d-8d50-42757a5c3534.jpg) | ![KakaoTalk_20220421_103825891_02](https://user-images.githubusercontent.com/64190071/164358185-a63371d7-84ad-4eb9-8337-c70857c0e170.jpg) | ![KakaoTalk_20220421_103825891_01](https://user-images.githubusercontent.com/64190071/164358129-a9ce91f8-84c5-4a9c-8329-27cf18e68e7f.jpg) |

## 문제 정의(대회소개) & Project Overview

![Untitled 2](https://user-images.githubusercontent.com/64190071/164357707-420bb60c-74f3-4aba-946f-a47dbc9edc24.png)

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## Dataset

- 11 class : BackGround, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing

## METRICS

- mIoU

![image](https://user-images.githubusercontent.com/64190071/170266765-c9456dd6-0f3a-447d-84dc-d8f3613cf2af.png)

### TOOLS

- MMSegmentation
- Github (Custom Git-flow Branching, Issue, PR)
- Notion
- Slack
- Wandb

### Models
- FCN
- FCN8s
- DeepLabV1
- DeepLabV3
- DeepLabV3Plus
- DeconvNet
- SegNet
- UperNet
- KNet
- OCRNet
- SegFormer
- FPN

### Backbones
- ResNet
- VGG16
- Xception
- Swin Transformer
- HRNet
- ConvNeXt
- Beit

### Schedulers
- Step
- Cosine Annealing Restart

### Augmentations
- ShiftScaleRotate
- GridDistortion
- Mosaic
- TTA (Multi-scale Testing, Flip)

### Data Cleansing
```
1. 봉투 안에 annotation 너무 상세하 되어있는거는 빼야한다.
2. 아예 클래스 Annotation을 잘못한 경우는 폐기
3. 애매하다 싶은거를 그냥 일반쓰레기로 한 경우 폐기
4. 담배곽 -> 일반쓰레기 아니면 폐기
5. 명함이나, 전단지 같은게 일반쓰레기가 아닌경우는 체크하고 내일 얘기하자
6. 가려진 것들을 지웠는지? 아니면 나뒀는지 애매한 것들은 체크하기
7. 기타
```
<img width="816" alt="image" src="https://user-images.githubusercontent.com/76461625/171345334-0d593dce-35d0-4d95-a856-56bc75ed1833.png">


## Result
### LB Score
|Public   | Private  |
|:-:|:-:|
| 0.7985  |    0.7406 |

## Citation

### MMDetection
```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
