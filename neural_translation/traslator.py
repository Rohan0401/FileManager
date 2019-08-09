from __future__ import absolute_import, division, print_function, unicode_literals
from .utils import preprocess_sentence, tokenize

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time


class DataProcessor:

  def __init__(self, path, num_examples):
    self.path = path
    self.nun_examples = num_examples

  def create_dataset(self):
    lines = io.open(self.path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:self.num_examples]]

    return zip(*word_pairs)

  def load_dataset(self):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = self.create_dataset(self.path, self.num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class CreateDataset:

  def __init__(self, input_tensor_train, target_tensor_train, buffer_size, batch_size, steps_per_epoch, embedding_dim,
               units, vocal_input_size, vocab_tar_size):
    self.input_tensor_train = input_tensor_train
    self.target_tensor_train = target_tensor_train
    self.buffer_size = buffer_size
    self.batch_size = batch_size
    self.steps_per_epoch = steps_per_epoch
    self.embedding_dim = embedding_dim
    self.units = units
    self.vocal_input_size = vocal_input_size
    self.vocab_tar_size = vocab_tar_size

  def generate_dataset(self):
    dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.target_tensor_train)).shuffle(
      self.buffer_size)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)

    return dataset


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
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


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


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


class Translator:

  def __init__(self, optimizer, loss_object, checkpoint_dir):
    self.optimizer = optimizer
    self.loss_object = loss_object
    self.checkpoint_dir = checkpoint_dir
    self.encoder = Encoder
    self.decoder = Decoder

  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

  def create_checkpoint(self):
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "cpkt")
    checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                     encoder=self.encoder,
                                     decoder=self.decoder)
    checkpoint.save(file_prefix=self.checkpoint_dir)

  @tf.function
  def train_step(self, inp, targ, batch_size, targ_lang, enc_hidden, encoder, decoder):
    loss = 0

    with tf.GradientTape() as tape:
      enc_output, enc_hidden = encoder(inp, enc_hidden)

      dec_hidden = enc_hidden

      dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)

      # Teacher forcing - feeding the target as the next input
      for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        loss += self.loss_function(targ[:, t], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    self.optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

  def perform_traning(self, epochs, inp, tar, dataset, steps_per_epoch):

    for epoch in epochs:
      start = time.time()

      enc_hidden = self.encoder.initialize_hidden_state()
      total_loss = 0

      for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = self.train_step(inp, tar, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
          self.create_checkpoint()

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  def evaluate(self, sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = self.encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
      predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                           dec_hidden,
                                                           enc_out)

      # storing the attention weights to plot later on
      attention_weights = tf.reshape(attention_weights, (-1,))
      attention_plot[t] = attention_weights.numpy()

      predicted_id = tf.argmax(predictions[0]).numpy()

      result += targ_lang.index_word[predicted_id] + ' '

      if targ_lang.index_word[predicted_id] == '<end>':
        return result, sentence, attention_plot

      # the predicted ID is fed back into the model
      dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
