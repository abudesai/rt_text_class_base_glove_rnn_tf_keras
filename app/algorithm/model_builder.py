from algorithm.preprocess.preprocess import prep_TEXT
import config
import numpy as np
import sys
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.layers import (
    Dense,
    Bidirectional,
    GRU,
    Flatten,
    Embedding,
    Input,
    GlobalMaxPooling1D,
)
import tensorflow as tf
import os
import logging
import os
import random

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


MODEL_NAME = config.MODEL_NAME
MODEL_SAVE_PATH = config.MODEL_SAVE_PATH
PRETRAINED_EMBEDD_PATH = config.PRETRAINED_EMBEDD_PATH
EMBED_DIM = config.EMBED_DIM

def set_seeds(): 
    seed_value = config.RAND_SEED
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


class RNN_pretrained_embed:
    def __init__(self):
        pass

    def __build_model_compile(
        self, num_y_classes, num_layers, neurons_num, embed_lay_output, learning_rate
    ):
        """This fucntion builds the model and compile it"""
        # tensorflow hub universal sentence encoder
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        metrics = [Precision(), Recall()]

        text_vectorizer = prep_TEXT.load_text_vectorizer()

        voc = text_vectorizer.get_vocabulary()

        embedding_dim = EMBED_DIM
        num_tokens = len(voc) + 2
        embed_layer = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(
                self.get_trained_embedd_matrix(voc=voc, embed_dim=embedding_dim)
            ),
            trainable=False,
            name="Embedding_Layer",
        )

        model = tf.keras.Sequential()
        model.add(Input(shape=(1,), dtype=tf.string))
        model.add(text_vectorizer)
        model.add(embed_layer)

        for i in range(num_layers):
            model.add(
                Bidirectional(
                    GRU(neurons_num, return_sequences=True),
                    name=f"Bidirectional_layer_{i}",
                )
            )

        model.add(GlobalMaxPooling1D())

        if num_y_classes > 2:
            model.add(Dense(num_y_classes, activation="softmax"))
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        else:
            model.add(Dense(1, activation="sigmoid"))
            loss = tf.keras.losses.BinaryCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # model.summary(); sys.exit()
        return model

    def fit(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=10,
        num_layers=1,
        neurons_num=50,
        embed_lay_output=120,
        learning_rate=1e-2,
    ):
        set_seeds()

        num_classes = len(np.unique(y_train))

        self.model = self.__build_model_compile(
            num_y_classes=num_classes,
            num_layers=num_layers,
            neurons_num=neurons_num,
            embed_lay_output=embed_lay_output,
            learning_rate=learning_rate,
        )
        if num_classes > 2:
            y_train = tf.squeeze(tf.one_hot(y_train, num_classes))
            if not y_val is None:
                y_val = tf.squeeze(tf.one_hot(y_val, num_classes))

        
        callbacks = [] # we add desired call backs in a list # If no callback is desired, leave list empty
        # Kindly change patience as you see fit in your project
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=2,
                verbose=1,
                restore_best_weights=True,
                monitor="loss" if x_val is None else "val_loss"
                )
            ) 

        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=None if x_val is None else (x_val, y_val),
            validation_steps=None if x_val is None else len(x_val),
            callbacks=callbacks,
            verbose=1
        )

        return self.model

    def get_trained_embedd_matrix(self, voc, embed_dim, trained_embedd_path=None):

        pretrain_embed_path = (
            pretrain_embed_path if trained_embedd_path else PRETRAINED_EMBEDD_PATH
        )

        embeddings_index = {}
        max_words = 100000
        with open(pretrain_embed_path) as f:
            for i, line in enumerate(f):
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

                if i == max_words:
                    break

        print("Found %s word vectors." % len(embeddings_index))

        num_tokens = len(voc) + 2
        embedding_dim = embed_dim
        hits = 0
        misses = 0
        word_index = dict(zip(voc, range(len(voc))))

        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1

        return embedding_matrix

    def save_model(self, save_path=MODEL_SAVE_PATH):
        path = os.path.join(save_path, MODEL_NAME)
        self.model.save(path)


def load_model(save_path=MODEL_SAVE_PATH):
    path = os.path.join(save_path, MODEL_NAME)
    model = tf.keras.models.load_model(path)
    print(f"Loaded model from: {path} successfully")
    return model