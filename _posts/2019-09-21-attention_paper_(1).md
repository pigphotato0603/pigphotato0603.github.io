---
layout: post
title: " attention papers(1)- Bahdanu&Luong attention "
---

## Attention - Bahdanau

1. intro

	기존의 NMT를 위한 encoder-decoder model에서는 원문장을 읽는 encoder의 input길이가 고정되어 있다. 때문에 원 문장의 모든 정보를 하나의 벡터인 Context vector에 압축시켜 넣어야하고, 이는 문장의 길이가 길어질 수록 많은 원문장의 정보를 손실시키게 된다.
    
    이러한 문제를 해결하기 위해, 본 논문은 '배열과 번역'을 jointly하게 진행하는 encoder-decoder model을 제시한다. 모델이 단어 각각을 번역하는 과정에서,  input단어들 중 가장 목표문장의 단어 하나를 생성하는데 큰 연관이 있는 정보를 찾아 각각에 가중치를 부여한다. 입력으로 들어가는 원문장을 벡터의 시퀀스로 encode하고, decode할 때 이 벡터 중 가장 적절한 부분집합을 선택하여 사용한다.  
    
2. 기존의 RNN encoder-decoder model
	
    Encoder: 입력문장(벡터의 sequence)를 통해 context vector C를 생성함. 다시 말하자면, C 벡터를 생성하기 위해 문장의 모든 요소(정확히 말하자면 모든 input단어들에 대한 은닉층-hidden state의 시퀀스를 한번에 비선형함수에 넣어 생성) 를 이용함.
    
3. Learning to align and translate- 제안모델
	
    * 기존의 RNN encoder-decoder model과의 차이점은, context vector을 만들 때 hidden state에 가중치 alpha를 각각 내적하여 그 값을 합쳐서 만든다는 것입니다. 
    * decoder부분: context vector Ci생성부분(i=output 단어 index) 을 설명하자면 e(i,j)=a[S(i-1),h(j)]로 구성됩니다. a는 alignment function으로 주로 dot product나 tanh를 사용합니다. 따라서 결과값은 스칼라가 나오며, 하나의 목표단어에 대하여 인풋단어의 수 j개 만큼의 스칼라가 나오고, 소프트맥스 함수를 취해 normalize해줍니다. 이것이 "attention 가중치"가 됩니다. 이 부분은 Feed forward NN(rnn과 달리 ,network에 순환이나 루프가 없음. 가장 단순한 형태의 nn으로 퍼셉트론 구조가 그 예이다.) 으로 구성되어 있어 추가적 연산을 필요로 합니다. 
    * 소프트맥스를 취해준 최종 가중치와 encoder의 hidden state(hj)와의 dot product를 한 결과가 "목표단어 yi에 대한 문맥벡터 Ci"가 됩니다. ci는 decoder의 hidden state si를 생성하는데 사용됩니다. 
    * encoder: bidirectional RNN을 이용해 양방향 모두의 정보를 습득 
 
4. 실험 모델

	* 모델을 2번씩 학습(최대 30단어로 이루어진 문장, 최대 50단어 문장으로 학습)
	* bi-directionRNN, 1000개의 hidden units
	* single maxout 은닉층, multilayer network를 이용해 목표단어에 대한 조건부확률 계산
	* minibatch sgd와 adadelta적용: 80문장씩 이루어짐. 5일 소요
	* 모델의 학습 후 beam search로 조건부확률을 최대화시키는 번역을 찾음
	
5. 결과 분석: BLEU score

## Effective Approaches to Attention-based NMT(Luong)

1. intro
	
    * global attention, local attention
    	local attention은 hard, soft attention을 섞은 방법으로, global보다 연산이 적고, 미분가능
        
    
2. attention-based models
	
    * 해당 논문은 attention방법을 global,local로 2가지를 제시합니다. : C(t)를 생성할 때 어느 부분을 고려하는가의 차이로 구분
    - global: attention이 모든 원 문장의 단어에 적용
    - local: attention이 원 문장의 부분집합에만 적용
    
    * common
    	- stacking LSTM 구조
    	
3. global attention-based model과 local attention-based model 비교

	(1) global attention-based model
    
    	* 
    
    
    
    
    
    
    
    
    