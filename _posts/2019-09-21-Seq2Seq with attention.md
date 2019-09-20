---
layout: post
title: "attention을 이해해보자(2)- Seq2Seq with attention model"

---
이 문서는 [https://github.com/graykode/nlp-tutorial](https://github.com/graykode/nlp-tutorial) 의 코드를 참고하였습니다. 

* prepare

```
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.reset_default_graph()
```

* datas

 S: 디코딩 인풋이 시작됨을 알려줍니다. 
 E: 디코딩 아웃풋이 끝남을 알려줍니다. 
 P: current batch data size가 time steps보다 짧을 때 그 공란에 P를 채워줍니다. 
 
```
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # vocab list


```

* parameters

~~~
n_step = 5  # 한 문장에 있는 최대 단어의 수(=number of time steps에 해당하는 숫자)
n_hidden = 128
~~~

* make_batch function 

~~~
def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return input_batch, output_batch, target_batch

~~~