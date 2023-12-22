# DataCapstone
표준어 텍스트를 입력으로 받아 경상도 방언을 출력하는 프로젝트입니다.

1. 먼저 T2T Model(KoUL2)이 표준어 텍스트를 경상도 방언 텍스트로 바꿔줍니다.
2. 이후 TTS Model(GlowTTS)이 경상도 방언 텍스트를 경상도 방언 스피치로 바꿔줍니다.

## Installation
```
pip install -r requirements.txt
```

## TTS Model
```
cd T2T
```

# Data Capstone
표준어 텍스트를 입력으로 받아 경상도 방언을 출력하는 프로젝트입니다.

1. 먼저 T2T Model(KoUL2)이 표준어 텍스트를 경상도 방언 텍스트로 바꿔줍니다.
2. 이후 TTS Model(GlowTTS, HifiGan)이 경상도 방언 텍스트를 경상도 방언 스피치로 바꿔줍니다.

## Installation
```
pip install -r requirements.txt
```

## TTS Model
```
cd TTS
```
### Raw Data 출처
filelists.zip 데이터 셋을 만들기 위한 아래의 Raw 데이터들을 다운받아줍니다.

<strong>AI HUB 경상도 사투리 데이터</strong>

- [한국어 방언 발화(경상도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=119)

<strong>Youtube 경상도 사투리 데이터</strong>

1) 경상도 어린왕자 오디오 1,2,3편
- https://youtu.be/4rQtdphNhD4?si=eblTnZ9Nq4wxrgFV
- https://youtu.be/SsSmaZgXOtg?si=vbxa0_Ad8MYvW3_H
- https://youtu.be/A9ZmcZ1pAnk?si=khNOw_9npTkA74lY

2. [native 사투리 동화 걸리버여행기](https://youtu.be/1XtUcImoshA?si=UBG_XEnGM64RwoBk)

3. [native 사투리 동화 피터팬](https://youtu.be/yKAtALhzpco?si=ge5Vd4D_coJtMuXe)

4. 경상도어 능력고사
- https://youtu.be/sPCY27BP0G4?si=0zCuwi-hTjuoQsb-
- https://youtu.be/64ovYJn5sC4?si=jZTKkkhfi-77oePM

5. [어린왕자 경상도버전 낭독](https://youtu.be/ic0SWQIgy2E?si=nR61ZYqtkUiP1_mD)

### Data Preprocessing
각 Raw 데이터에 대하여 1 -> 2 순으로 전처리 및 filelists 폴더를 만들어주는 과정을 진행합니다.

- AI HUB 데이터 전처리
1) `AIHUB_data_preprocessing.ipynb`
2) `make_filelists.py`

- Youtube 데이터 전처리
1) `YoutubeData_preprocessing.ipynb`
2) `make_filelists.py`

### Train
모든 TTS 모델 학습은 Google Colab에서 T4 GPU와 고용량 메모리를 사용하여 진행했습니다. 메모리 이슈로 학습이 안될 경우 배치사이즈 및 데이터 길이를 적절히 조절해주세요. 
또한, 마운트 할 구글 드라이브 내에 다음 파일이 존재하는지 꼭 확인해주세요.
- `/Colab Notebooks/data/filelists.zip`

<strong>Fine-Tuning Glow-TTS</strong>
- `train_glow-tts.ipynb`

<strong>Fine-Tuning Hifi-GAN</strong>
- `train_hifi-gan.ipynb`

### Inference
Glow-TTS 모델과 Hifi-GAN 모델을 같은 filelists.zip 데이터로 각각 Fine-Tuning 한 후, 새로 만들어진 모델 폴더 경로를 inference 할 때 사용합니다. 자세한 내용은 아래의 파일에서 확인해주세요.
- `inference.ipynb`



## T2T Model
```
cd T2T
```

### Data Preprocess
먼저 aihub의 [한국어 방언 발화(경상도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=119) 데이터를 다운받습니다. `T2T` 디렉토리 내에 `gyeongsang_dialect_data` 디렉토리를 만들어 json 파일과 txt 파일을 해당 디렉토리 내에 저장합니다. 이 때 wav 파일은 저장할 필요 없습니다. 이후 아래 커맨드를 실행하면 `gyeongsang_dialect_csv` 디렉토리가 생성되며, `same_forms.csv` 파일과 `different_forms.csv` 파일이 생성됩니다.
```
python make_csv.py
```

### Train
모든 T2T 모델 학습은 RTX 3090 1ea 환경에서 진행했습니다. 메모리 이슈로 학습이 안될 경우 적절히 배치사이즈를 조절해주세요. 
`YOUR_MODEL_PATH`에 저장할 폴더를 지정해주세요.

FineTuning KoBART
```
python train_bart.py \
--pretrained_model_name_or_path gogamza/kobart-base-v2 \
--save_dir $YOUR_MODEL_PATH \
--save_strategy epoch \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--use_wandb \
--entity dannykm \
--wandb_model_name kobart_test \
--project_name dialect \
```
FineTuning KoUL2 (Recommend)
```
python train_ul2.py \
--pretrained_model_name_or_path DaehanKim/KoUL2 \
--save_dir $YOUR_MODEL_PATH \
--save_strategy epoch \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 5 \
--use_wandb \
--entity dannykm \
--wandb_model_name ul2_test \
--project_name dialect \
```
### Inference
```
python inference.py \
--model_name_or_path $YOUR_MODEL_PATH
```
