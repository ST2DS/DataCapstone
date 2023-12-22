###################################################
'''
AIHUB_data_preprocessing.ipynb 혹은 YoutubeData_preprocessing.ipynb로
raw data를 전처리 및 데이터에 대한 metadata.txt 파일을 만든 다음에 이 파일을 실행한다. 

make_filelists.py는 학습용 데이터 형식에 맞춰 filelists 폴더 구성을 만드는 과정임.
'''
###################################################

import os
import sys
import json
from pathlib import Path
import random
from unicodedata import normalize
from pydub import AudioSegment

# USER_UUID = sys.argv[1]

# audio_path = Path('.') / 'audio_files' / USER_UUID
# ljs_path = Path('.') / 'filelists'
# wav_path = ljs_path / 'wavs'

# ljs_path.mkdir()
# wav_path.mkdir()

audio_data = []

# print(f"audio_path: {audio_path}")
# print(f"ljs_path: {ljs_path}")

current_path = "/Users/soohyun/Downloads/" # filelists 폴더를 둔 경로
audio_input_path = "/Users/soohyun/Downloads/New_Sample/processed_data/" #전처리한 음성데이터 경로
audio_input_path = "/Users/soohyun/Downloads/New_Sample/wavs/" # 이 파일에서 추가 전처리 후 새로 음성데이터 저장할 경로
label_input_path = "/Users/soohyun/Downloads/New_Sample/label/"
woman_path = "/Users/soohyun/Downloads/speaker_woman/" # 여성 화자들의 음성데이터만 모아둔 폴더 경로
man_path = "/Users/soohyun/Downloads/speaker_man/" # 남성 화자들의 음성데이터만 모아둔 폴더 경로
output_path = "/Users/soohyun/Downloads/filelists/" # 최종적으로 학습시킬 데이터와 레이블링 마치고 train/test 데이터 나눠 저장할 폴더 경로

sr = 22050 # sampling rate


# make filelists
Output_path = [woman_path, man_path]

for path in Output_path:
    # make filelists
    with open(f'{path}metadata.txt', 'r', encoding='utf8') as metadata:
        for line in metadata:
            if not line:
                continue
            data = line.split('|')
            data[1] = data[1][:-1] # 문장끝에 \n 부분 잘라내기

            original_wav_path = current_path + data[0]
            
            data[0] = data[0].split("/")[-1] # wav 파일명만 추출 (filename.wav)

            print(f"Converting {original_wav_path}...")

            sound = AudioSegment.from_wav(original_wav_path)
            sound = sound.set_channels(1)
            sound = sound.set_frame_rate(22050)
            sound.export(output_path + "wavs/" + data[0], format="wav")

            audio_data.append(data)

    random.shuffle(audio_data)
    splitter = int(len(audio_data) * 0.8)

    print(f"Generating ljs_audio_text_train_filelist.txt...")
    with open(path + 'ljs_audio_text_train_filelist.txt', 'w', encoding='utf8') as filelist:
        for audio_item in audio_data[:splitter]:
            original_wav_filename = audio_item[0]
            text = audio_item[1]

            filelist.write(f'filelists/wavs/{original_wav_filename}|{text}\n')

    print(f"Generating ljs_audio_text_val_filelist.txt...")
    with open(path + 'ljs_audio_text_val_filelist.txt', 'w', encoding='utf8') as filelist:
        for audio_item in audio_data[splitter:]:
            original_wav_filename = audio_item[0]
            text = audio_item[1]

            filelist.write(f'filelists/wavs/{original_wav_filename}|{text}\n')

    print(f"Generating ljs_audio_text_test_filelist.txt...")
    with open(path + 'ljs_audio_text_test_filelist.txt', 'w', encoding='utf8') as filelist:
        pass

    print(f"Generating metadata.csv...")
    with open(path + 'metadata.csv', 'w', encoding='utf8') as filelist:
        for audio_item in audio_data:
            original_wav_filename = audio_item[0]
            text = audio_item[1]
            filename = original_wav_filename[:-4]

            filelist.write(f'{filename}|{text}\n')