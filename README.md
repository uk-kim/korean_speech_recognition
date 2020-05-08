# Korean Speech Recognition

한국어 음성인식을 목적으로 공부한다.

## 1. Introduction
CTC 방법을 토대로 음향모델을 학습하고, 향후 언어모델(방법은 현재 미정)을 적용하고자 한다.

음향모델에서는 오디오를 입력(MFCC)받아 발음 음소 단위의 시퀀스 출력을 찾는 목적의 학습을 수행한다.

## 2. 한국어 학습 데이터
* 한국어 학습 데이터는 AIhub에서 제공하는 음향데이터를 토대로 진행하고자 한다.
* 음향데이터의 transcript를 발음 형태로의 변환은 [ko g2p](https://github.com/scarletcho/KoG2P)를 통해 수행한다.

## 3. Data Preparation
### Transcript
  * [KoG2P](https://github,.com/scarletcho/KoG2P)를 활용하여 주어진 transcript를 발음 형태로 변환한다.
#### Example
  * AIhub 데이터셋의 transcript 파일은 euc-kr 포멧으로 인코딩 되어 있다.
```
with open('./sample/sample.txt', 'r', encoding='euc-kr') as f:
    lines = f.readlines()
print(lines)
```
  * `./sample/sample.txt`:  '아/ 몬 소리야, 그건 또. b/\n' 

### Audio feature transform
  * `python-speech-features` 패키지를 통한 mfcc feature 사용
### Code
  * transcript 와 audio 데이터를 불러와 각각 label encoding, feature transform을 수행하여, [file_name, labels, feature] 형태의 리스트를 `pickle` 포멧으로 저장한다.
    - 향후 이 부분은 수정해서 tf_record 형태로 변환해도 될듯.
	- 주어진 데이터셋은 오디오 길이가 모두 다르기 때문에 향후 배치 학습을 위해서 길이가 다른 파일들에 대해 padding을 적용하는 부분이 필요함.
  ```
  python prepare_dataset.py --datadir {directory where your own data is in} --outdir {path where preprocess data will be save as pickle type}
  ```

--- 


자세한 내용은 지속적으로 공부하며 적용할 예정이다.
