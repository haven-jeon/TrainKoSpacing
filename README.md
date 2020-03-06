
- [Automatic Korean word spacing with neural n-gram detector(NND)](#automatic-korean-word-spacing-with-neural-n-gram-detectornnd)
	- [Introduction](#introduction)
	- [Architecture](#architecture)
	- [Performances](#performances)
	- [How to Run](#how-to-run)
		- [Installation](#installation)
		- [Dependencies](#dependencies)
		- [Data](#data)
			- [Format](#format)
		- [Requirement](#requirement)
		- [Training](#training)
		- [Evaluation](#evaluation)
	- [Citation](#citation)
  

# Automatic Korean word spacing with neural n-gram detector(NND)

We propose a Korean word spacing method using Convolution based Neural N-gram Detector(NND) model for the problem of word spacing Korean sentences that have been completely removed. The NND model solves the disadvantages of the existing RNN model and can achieve high performance in the sequence labeling problem of natural language processing. Experimental results show that the NND model is 98.02% accuracy.
This model strongly related (Py)KoSpacing model[^2], but some points was improved.

## Introduction

- Maximize automatic Korean word spacing performance by using parallel convolution filters as a tool for extracting syllable n-gram probability information.
- This model was constructed using five NND(Neural N-gram Detector)s.
- Better performance than Bi-Directional LSTM(or GRU) based spacing engine under the same conditions.

## Architecture

![kosapcing_img](img/kosapcing_img.png)

## Performances

- Training Set : UCorpus-HG train set (846,930)
- Test Set :  UCorpus-HG test set (94,103)

We used four evaluation measures: character-unit precision ($P_{char}$), word-unit recall ($R_{word}$), word-unit precision ($P_{word}$), and word-unit F1-measure ($F1_{word}$).


| Algorithm                     | $P_{char}$ | $P_{word}$ | $R_{word}$ | $F1_{word}$ |
| ------------------ | ---------- | ---------- | ---------- | ---------- |
| CRF[^1]       | 0.9337     |     0.8471       |   0.8481         |   0.8476       |
| Bi-Directional GRU   |   0.9689    |    |     |      |
| Bi-Directional GRU with CRF     |    0.9717    |            |            |            |
| NND only      |   0.9726     |     0.9588   |    0.9591   |    0.9589    |
| NND(with  Bi-Directional GRU)      | 0.9802     |   0.9787    |    0.9788     |    0.9787        |

- $P_{char}$ = # correctly spaced characters/# characters in the test data.
- $R_{word}$ = # correctly spaced words/# words in the test data.
- $P_{word}$ = # correctly spaced words/# words produced by the system.
- $F1_{word}$ =$2 \times \frac{  P_{word}  \times R_{word}} {(P_{word} + R_{word})}$

*UCorpus-HG from [2017 Exobrain corpus](http://aiopen.etri.re.kr/service_corpus.php)*

## How to Run


### Installation

- For training, a GPU is strongly recommended for speed. CPU is supported but training could be extremely slow.
- Support only above Python 3.6.

### Dependencies

```
pip install -r requirements.txt
```

### Data

We mainly focus on the Sejong corpus, and the code takes its altered format as input. However, due to the license issue, we are restricted to distribute this dataset. You should be able to get it [here](http://aiopen.etri.re.kr/service_corpus.php).

#### Format

Bziped file consisting of one sentence per line.

```
gogamza@192.168.1.1:~/KoSpacing/data$ bzcat UCorpus_spacing_train.txt.bz2 | head
엠마누엘 웅가로 / 의상서 실내 장식품으로… 디자인 세계 넓혀
프랑스의 세계적인 의상 디자이너 엠마누엘 웅가로가 실내 장식용 직물 디자이너로 나섰다.
웅가로는 침실과 식당, 욕실에서 사용하는 갖가지 직물제품을 디자인해 최근 파리의 갤러리 라파예트백화점에서 '색의 컬렉션'이라는 이름으로 전시회를 열었다.
```

### Requirement

- mxnet (>= 1.0)
- tqdm (>= 4.19.5)
- pandas (>= 0.22.0)
- gensim (>= 3.8.1)

### Training

```{python}
python train.py --train --train-samp-ratio 1.0 --num-epoch 20  
```

### Evaluation

```{python}
python train.py --model-params model/kospacing.params
sent > 중국은2018년평창동계올림픽의반환점에이르기까지아직노골드행진이다.
중국은2018년평창동계올림픽의반환점에이르기까지아직노골드행진이다.
spaced sent > 중국은 2018년 평창동계올림픽의 반환점에 이르기까지 아직 노골드행진이다.
```

## Citation

```markdowns
@misc{heewon2018,
author = {Heewon Jeon},
title = {Automatic Korean word spacing with neural n-gram detector},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/haven-jeon/Train_KoSpacing}}
```

[^1]: https://github.com/scrapinghub/python-crfsuite
[^2]: https://github.com/haven-jeon/PyKoSpacing
