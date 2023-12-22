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

## T2T Model
```
cd T2T
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
