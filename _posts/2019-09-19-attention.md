---
layout: post
title: " attention "
comment: true

---

## 이미지 삽입 test

![](https://brunch.co.kr/@bong/814)

## 1. introduction

attention은 Neural Machine Translation과 다른 NLP 문제들을 해결하는데 state-of-art results를 만들어 내면서, 특히 word embedding 과 합쳐지면서 그 성능을 입증했습니다. attention 메커니즘은 다른 알고리즘, 예를 들면 BERT와 같은 알고리즘의 한 부분입니다. attention network에서 우리는 **transformer** 이란 한 종류의 네트워크를 만듭니다. 만약 transformer을 이해한다면, attention을 쉽게 이해할 수 있습니다. 그리고 transformer을 이해하기 위한 가장 best한 방법은 이전의 신경망과 비교대조를 하는 것입니다. transformer와 기존의 NN은 여러모로 다른데, 특히 input을 처리하는 프로세스와 input을 relevant features와 recombine하는 프로세스에서 다릅니다. 

## 2. Feed-forward networks

feed-forward network, 다시말해 vanila neural network(like a multilayer perceptron with fully connected lyaers)을 생각해봅시다. feed-forward network를 줄여서 ffn 이라 부르겠습니다. ffn은 모든 input features를 unique하고 독립적 존재로 인식합니다.

예를 들어, 개인들에 대한 데이터를 encoding한다고 가정합시다. 그리고 network에 feeding할 features는 age,gender,zip code, height, last degree, profession, political affiliation, number of siblings 라 합시다. 각 피쳐들에 대하여, 우리는 자동적으로 어떤 피쳐와 어떤 피쳐가 `서로 근접해있다`고 판단할 수 없습니다. 

age에서 gender로 바로 도약할수 있는, 즉 연결할 수 있는 가정을 만드는 일은 쉬운 일이 아닙니다. 

## 3. convolutional networks

이미지를 input으로 받아온다고 가정합시다. object에 대한 반사는 존재합니다. 예를 들면, 보라색 플라스틱 커피머그컵이 있다면, 머그컵을 구성하는 각 원자는 보라색 플라스틱 원자와 강한 연관성이 있을 것입니다. 이러한 것들은 픽셀로서 나타날 것입니다.

그래서, 만약 우리가 하나의 보라색 픽셀을 볼 수 있다면, 그것은 우도를 증가시킬 것입니다. 우도란, 다른 보라색 픽셀이 우리가 처음 발견한 하나의 보라색픽셀 근처에 다양한 방향으로 있을것이라는 확률을 말합니다.
게다가, 보라색플라스틱커피머그컵은 더 큰 이미지로써 공간을 차지할 것인데, 