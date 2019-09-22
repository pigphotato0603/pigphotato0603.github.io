---
layout: post
title: "attention을 이해해보자(2)- Seq2Seq with attention model"

---
이 문서는 [https://github.com/graykode/nlp-tutorial](https://github.com/graykode/nlp-tutorial) 의 코드를 참고하였습니다. 

* prepare

~~~
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.reset_default_graph()
~~~

* datas

 	S: 디코딩 인풋이 시작됨을 알려줍니다. 
 
	 E: 디코딩 아웃풋이 끝남을 알려줍니다. 
 
 	P: current batch data size가 time steps보다 짧을 때 그 공란에 P를 채워줍니다. 
 
 
~~~
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

	# sentences의 문장들을 구성하는 vocab들을 나눈다.
word_list = " ".join(sentences).split()
	# set으로 변경하고 다시 리스트로 변경: 중복 vocab없앰
word_list = list(set(word_list))
	# 
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # total vocab list=11개 단어 있음. 

~~~

* parameters

~~~
n_step = 5  # 한 문장에 있는 최대 단어의 수(=number of time steps에 해당하는 숫자)
n_hidden = 128
~~~

* make_batch function 

~~~
def make_batch(sentences):
	# 아래 파라메터들은 리스트인데 안에 (1*11)array가 들어있음
    # 각 array한개는 한 단어(vocab)가 벡터로 변형된 것
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]
    return input_batch, output_batch, target_batch

~~~

* model

~~~
enc_inputs = tf.placeholder(tf.float32, [None, None, n_class])  # [batch_size, n_step, n_class]
dec_inputs = tf.placeholder(tf.float32, [None, None, n_class])  # [batch_size, n_step, n_class]
targets = tf.placeholder(tf.int64, [1, n_step])  # [batch_size, n_step], not one-hot

~~~

* linear for attention
`tf.squeeze(input,axis): input은 tensor, returns a tensor of the same type with all dimensons of size 1 removed. 즉 차원 중 사이즈=1인 차원을 제거한다. 

`tf.reshape()에 대한 정보`: [https://www.guru99.com/tensor-tensorflow.html](https://www.guru99.com/tensor-tensorflow.html)

~~~
# Linear for attention
## attn: encoder's output time sequence
attn = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
out = tf.Variable(tf.random_normal([n_hidden * 2, n_class]))

def get_att_score(dec_output, enc_output):  # enc_output [n_step, n_hidden]
    score = tf.squeeze(tf.matmul(enc_output, attn), 0)  # score : [n_hidden]
    # dec_output을 0과 1 dim만 squeeze한다. 
    dec_output = tf.squeeze(dec_output, [0, 1])  # dec_output : [n_hidden]
    return tf.tensordot(dec_output, score, 1)  # inner product make scalar value

def get_att_weight(dec_output, enc_outputs):
    attn_scores = []  # list of attention scalar : [n_step]
    enc_outputs = tf.transpose(enc_outputs, [1, 0, 2])  # enc_outputs : [n_step, batch_size, n_hidden]
    
    # 5개의 timestep 에 대한 scores 계산 
    for i in range(n_step):
        attn_scores.append(get_att_score(dec_output, enc_outputs[i]))

    # Normalize scores to weights in range 0 to 1
    return tf.reshape(tf.nn.softmax(attn_scores), [1, 1, -1])  # [1, 1, n_step]

~~~

* build model

```
model = []
Attention = []
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    # enc_outputs : [batch_size(=1), n_step(=decoder_step), n_hidden(=128)]
    # enc_hidden : [batch_size(=1), n_hidden(=128)]
    enc_outputs, enc_hidden = tf.nn.dynamic_rnn(enc_cell, enc_inputs, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    inputs = tf.transpose(dec_inputs, [1, 0, 2])
    hidden = enc_hidden
    for i in range(n_step):
        # time_major True mean inputs shape: [max_time, batch_size, ...]
        dec_output, hidden = tf.nn.dynamic_rnn(dec_cell, tf.expand_dims(inputs[i], 1),
                                               initial_state=hidden, dtype=tf.float32, time_major=True)
        attn_weights = get_att_weight(dec_output, enc_outputs)  # attn_weights : [1, 1, n_step]
        Attention.append(tf.squeeze(attn_weights))

        # matrix-matrix product of matrices [1, 1, n_step] x [1, n_step, n_hidden] = [1, 1, n_hidden]
        context = tf.matmul(attn_weights, enc_outputs)
        dec_output = tf.squeeze(dec_output, 0)  # [1, n_step]
        context = tf.squeeze(context, 1)  # [1, n_hidden]

        model.append(tf.matmul(tf.concat((dec_output, context), 1), out))  # [n_step, batch_size(=1),

```

* ?? model

```
trained_attn = tf.stack([Attention[0], Attention[1], Attention[2], Attention[3], Attention[4]], 0)  # to show attention matrix
model = tf.transpose(model, [1, 0, 2])  # model : [n_step, n_class]
prediction = tf.argmax(model, 2)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
```

* training &test model

```
# Training and Test
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(2000):
        input_batch, output_batch, target_batch = make_batch(sentences)
        _, loss, attention = sess.run([optimizer, cost, trained_attn],
                                      feed_dict={enc_inputs: input_batch, dec_inputs: output_batch, targets: target_batch})

        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    predict_batch = [np.eye(n_class)[[word_dict[n] for n in 'P P P P P'.split()]]]
    result = sess.run(prediction, feed_dict={enc_inputs: input_batch, dec_inputs: predict_batch})
    print(sentences[0].split(), '->', [number_dict[n] for n in result[0]])

```

* show attention

```
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()
```

