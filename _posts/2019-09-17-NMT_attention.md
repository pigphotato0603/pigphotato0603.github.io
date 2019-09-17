---
layout: post
title: "Neural MAchine Translation with Attention"

---

# NMT with Attention

## model의 개요
이 번역 모델은 spanish 를 english로 번역하는 seq2seq model을 train하는 것을 목적으로 합니다. input으로는 Spanish sentence가 들어가며, output으로는 English로 번역된 문장이 리턴됩니다. 
먼저, 필요한 패키지, 모듈을 깔아줍니다. 여기서는 여러가지 프래임워크 중 tensorflow를 사용하였습니다. 

## download and prepare the dataset
[http://www.manythings.org/anki/](http://www.manythings.org/anki/) 에서 제공하는 language dataset을 사용할 것입니다. 이 데이터셋은 번역 전 문장과 번역 후 문장이 서로 pair가 되게 저장되어 있습니다. 많은 데이터셋 중 우리는 English-Spanish dataset을 사용하겠습니다.  
데이터를 다운 받은 다음, 아래와 같이 4가지 스텝으로 preprocessing 해야합니다.
1. 각 문장에 *start*와 *end* token을 추가해줍니다.
2. special characters를 각 문장마다 제거해줍니다.
3. word index를 생성하고, reverse해줍니다.( dictionaries mapping from word -> id and id->word )
4. each sentence를 a maximum length가 되게 padding 해줍니다.



~~~~
# Converts the unicode file to ascii
# unicodedata module
# unicodedata.normalize(form,unistr):
# 유니코드 문자열 unistr에 대한 정규화형식 form을 반환, form의 값은 NFC,NFD.. 등이 있다. 

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
	# w를 소문자로바꾸고 양쪽 공백 지운다음 위의 함수에 적용시켜준다.
    # w는 한 문장단위이다.
    w = unicode_to_ascii(w.lower().strip())

    # 문장에서 단어와 . 사이 공백 하나를 생성한다.
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
	
    # 문장 내에 영어소문자,대문자, 기본 부호들 몇개 제외한 나머지는 모두 공백처리
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    # rstrip: 오른공백 지우기
    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

~~~~
주어진 코드가 제대로 작동하는지 아래의 예시를 주어 확인해보자.
~~~
en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))
~~~

이제는 악센트를 없에고, sentences를 clean하고, word pairs가 되게 만들어 줍니다.

**zip()함수에 대해 좀 알고 가 보자.**
```
p = [[1,2,3],[4,5,6]]
## zip함수는 같이 zip할 여러개의 arg를 input으로 원하는데, 여기서
#p 는 단일의 하나의 리스트이며, 리스트의 원소 역시 리스트이다. 즉 
# 1개의 리스트임. *를 안에 써주면, input리스트가 하나여도, input 리스트의 원소가 여러개의 리스트이면, 그 여러개의 리스트끼리 zip해준다는 것

p1=zip([1,2,3],[4,5,6]) # same as zip(*p)
p2=zip([[1,2,3],[4,5,6]]) # same as zip(p)
print(list(p1))
print(list(p2))
```

~~~
# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
# path에 있는 data
def create_dataset(path, num_examples):
# 한 줄 간격으로 문장을 구분한다. 
# lines는 리스트이며 리스트원소 갯수는 118964개의 문장을 가지고 있으며 영어, 스페니쉬 합쳐서 있음
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
# 페어로 저장된 리스
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    return zip(*word_pairs)

~~~
함수가 잘 작동되는지 확인합니다. 리턴값을 보면 start, end token이 제대로 뭍혀져 있으며 단어와 . 사이 공백이 존재합니다. 
~~~
en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])
~~~

## create a tf.data dataset

~~~
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
~~~

**next(iter(dataset)) function**
next 함수 안에 iterator 객체가 input으로 들어옴 보통 인풋의 dtype은 list임.


~~~
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape
~~~

## Encoder class

~~~
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

~~~

## Make Encoder class object
~~~
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

## sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
~~~

## Attention model -attention layer

~~~
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
~~~







