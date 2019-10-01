---
layout: post
title: "Attention papers(2)- Feed-forward& Disan attention"
---

## Feed-forward attention

[DSBA_korea.univ의 youtube참고하였습니다](https://www.youtube.com/watch?v=sfh-G-9LhOU&list=PLetSlH8YjIfUuwVM3j9XQ3UQTrY2KhdO1&index=19)

1. intro
	
    * RNN 기반 구조는 seqence가 길면 장기 메모리 문제 발생
    * 따라서 RNN기반 구조는 sequence길이가 수백개로만 제한이 됨
    * RNN 뿐만 아니라, feed-forward 신경망 구조에도 사용할 수 있는 attention이 필요함
    * Bahdanau의 attention을 간소화한 구조를 제시함

2. attention의 2가지 제안

	(1) feed-forward atention
    
    	* alignment function은 input sentence의 히든state인 h(t)에만 의존함
    	* context vector은 alpha를 가중치로 하는 ht에 대한 가중평균
    	* 위 2가지로 보았을 때, Bahdanau의 모델구조보다 간소화된 모델구조임을 알 수 있습니다. 이를 통해 은닉상태가 recurrent하지 않은 feed-forward에도 적용할 수 있게 되었습니다.
    	* feed-forward는 연산의 병렬화가 가능하기 때문에 훨씬 효율적인 구조여서 이 ffnn에 적용하는 것은 의미가 있습니다.
    	* 단, ffnn에 적용하기 위해서는, input sequence length가 고정되어 있어야 합니다. 길이가 만약 가변적이라면, temporal integration이 필요합니다. 이는 시점을 통합한다는 의미로, 시점정보를 없애겠다는 의미입니다. 이 때, context vector는 sum(h(t))로 구성되는데, t=1~length(input_x_sequence)입니다.


## Disan: Directional Self-attention Network for RNN/CNN-free Language Understanding

1. intro

	* RNN은 장기적 정보, CNN은 지엽적 정보를 잘 포착함
	*  attention은 시퀀스 내의 요소들의 거리와 관계없이, 요소 간 중요한 의존관계를 포착하기 위해 작동
	*  기존의 attention은 RNN/CNN과 같이 elements 간의 거리정보가 고려되는 모델의 보충적 역할에 그침
	*  이 Disan은 RNN/CNN 구조 없이, 오로지 attention으로만 문장 임베딩을 학습
	*  기존의 attention에서 시간순서에 대한정보 directional와 시퀀스에 대한 다차원정보를 처리할 수 있는 기능이 추가 

	* 복잡한 RNN모델 보다 높은 예측력과 시간효율성을 가짐
	* SNLI data를 통해 테스트한 결과, 비교한 다른 방법론들 보다 높은 정확도를 얻음
	* 다른 데이터에서도 최고의 성능을 보임
	
2. 구성

	* Disan은 두가지 요소가 더 첨가된 directional self-attention으로 구성됨( self-attention은 transformer과는 다른 구조임 )
	- multi-dimensional: 이전에는 단어들의 관계에 대한 attention이 vector형태였으나, matrix형태로 바꿔서 단어의 feature까지 고려할 수 있음
	- 여러가지 masking을 통해 단어 간의 비대칭적 attention을 형성하여 시작순서를 고려 가능하게 함. 
	
	`self attention이란 attention의 특이케이스로, 기존의 attention은 input sequence x(i)와 목표단어(쿼리)q 간의 attention을 보았다면, self-attention은 같은 input sequence 단어 x(i)와 x(j) 간의 attention을 계산함`  
    
3. 제안 모델

	* multi-dim attention
		- 언어의 다의성까지 고려하여 단어를 임베딩하기 위해, attention시 단어 feature에 대한 다원성 고려를 위해 scalar에서 multidimension(matrix)로 늘렸습니다. 
        - 자세히로는, alignment score산출과정에서 벡터였던 w를 행렬 W로 대체하며, bias term을 추가했습니다.
        
        -  따라서, f(x(i),q)가 단일 스칼라였다면, 이제는 R^(d-emb)차원으로 확장되었습니다.
        -  다차원 alignment score에 softmax를 적용하여 categorical distribution p(z(k)=i|x,q)를 산출합니다. 여기서 이 확률이 의미하는 바는, 목표단어 q와 input sequence x가 주어졌을 때, q를 예측하기 위해 사용되는 x(i)단어의 m가지 feature들(k=1~m)의 기여도를 확률로 나타낸 것입니다.  그렇다면 만약 2번째 features에 대한 확률이 높다면 q단어를 예측하기 위해 x(i)단어의 여러가지 특징 중 2번째 특징을 많이 참고해야한다는 뜻입니다.

	* multi-dim self-attention(token2token)
	
    	-  단순 multi-dim에 self-attention을 도입
    	-  alignment score 생성 과정에서 같은 문장 내(x)의 단어 x(i)와 x(j) 간의 의존도를 찾음( 앞에서 말했던,`self-attention은 같은 input sequence 단어 x(i)와 x(j) 간의 attention을 계산함` 이부분을 구현한 것이 됨 )
    	-  multi-dimensional attention의 식에서 q를 x(j)로 변경
    	
    * multi-dim self-attention(source2token)
    
     	- 앞의 token2token 모델에서는 input seqence단어 끼리의 의존도를 계산했다면, 이 모델은 input sequence단어 1개와 전체 input 문장 간의 의존관계를 찾음
     	- 단순 multi-dim attention에서 q를 제거하면 됩니다.
     	- p와 s에 대한 공식은 이전과 동일하게 진행됩니다.

3. Directional self attention(Disa)

	* 구성: 크게 fc layer, masked self-attention , fusion gate
	* fc layer: sequence x를 sequence of hidden state h로 변환
		- h=f(W*x + b )
		-  W.shape=(dim_hidden,dim_embedding_user)
		- x.shape=(dim_embedding_user,input_sequence_length)
		-   b.shape=(dim_hidden,input_sequence_length)  		 - h.shape=(dim_hidden,input_sequence_length)
		-   즉 x행렬의 각 열벡터는 인풋단어 하나에 대한 임베딩벡터
		-   그리고 아마 dim_embedding_user의 수는 앞서 말한 단어의 feature수일 것이다..
		-   이 작업으로 얻을 수 있는 효과: input size변형을 통한 차원축소의 효과(발표자 정민성 님의 의견)..

	* masked self-attention: token2token attention에 masking하여 의존성과 시간도를 고려
		- input seqence x 대신 h를 통해 self-attention 진행
		- h(i)와 h(j) 간의 유사도 측정
		- activation function을 tanh로 하고 c를 곱함, 함수 내 값들도 c로 나눔(연산효율)
		- positional masking: 요소 간 attention을 비대칭적으로 만듦(방향, 시간순서), 텀 하나가 위의 함수에 추가됨 
		- masking을 적용한 후, categorical distribution p를 산출함(각 단어마다 쿼리단어를 예측하는 데 가장 중요한 feature별로 확률이 측정됨) => score는 p를 h에 대한 가중평균을 통해 도출됨
		
	* fusion gate: attention 이전의 input과(h) 이후의 output을 결합함
		- s와 h를 통해 fusion-score를 산출함(시그모이드함수)
		- s와 h에 대해 F를 적용해서 결합하면 최종 결과물인 u를 도출
		- u는 fc 거친 input과 attenton적용결과의 결합
