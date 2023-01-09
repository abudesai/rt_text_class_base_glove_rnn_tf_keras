import numpy as np
import pandas as pd
from algorithm.preprocess.schema_handler import produce_schema_param
import config
import os
import pickle
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


ARTIFACTS_PATH = config.PREPROCESS_ARTIFACT_PATH
DATA_SCHEMA = config.DATA_SCHEMA
TEXT_VECTORIZER_NAME = config.TEXT_VECTORIZER_NAME


class preprocess_data:
    def __init__(
        self,
        data,
        data_schema=DATA_SCHEMA,
        artifacts_path=ARTIFACTS_PATH,
        shuffle_data=True,
        train=True,
        gen_val_data=True,
    ):
        """
        args:
            data: The data we want to preprocess
            data_schema: The schema that will handle the data
            artifacts_path: Defines the path of generated preprocess artifacts during training
            shuffle_data: If True it will shuffle the data before processing it
            train: if it's True it will save artifacts to use later in serving or testing
            gen_val_data: If True, will split data into train and validation data
        """
        if not isinstance(
            data, pd.DataFrame
        ):  # This should handle if the passed data is json or something else
            self.data = pd.DataFrame(data)

        else:
            self.data = data

        self.gen_val_data = gen_val_data
        self.data_schema = data_schema
        self.sort_col_names = []
        self.schema_param = produce_schema_param(self.data_schema)
        self.artifacts_path = artifacts_path
        self.train = train
        self.LABELS = self.define_labels()  # Get's labels columns
        self.id_col = ""
        self.clean_data()  # Checks for dublicates or null values and removes them

        if shuffle_data:
            self.data.sample(frac=1).reset_index(drop=True)

        self.fit_transform()  # preprocess data based on the schema
        self.sort_as_schem()
        if self.train:
            self.save_label_pkl()

    def clean_data(self):
        if self.data.duplicated().sum() > 0:
            self.data.drop_duplicates(inplace=True)

        self.data.dropna(inplace=True)

        self.data.reset_index(drop=True)

    def fit_transform(self):
        """preprocess data based on the schema, in case it's not training then it will load the preprocess pickle object"""
        for i, key in enumerate(self.schema_param.keys()):
            # for sorting the columns name later
            if i == 0:  # Ids column always the first key and first column
                self.id_col = self.schema_param[key]
            if key == "idField":
                # It does nothing, but in case we decided to do something in the future
                col_name = self.schema_param[key]
                self.sort_col_names.append(col_name)
                self.data[col_name] = prep_NUMERIC.handle_id(self.data[col_name])

            elif (
                key == "targetField"
            ):  # Will assume it's label and startes to label encode it
                if self.train:
                    col_name = self.schema_param[key]
                    self.sort_col_names.append(col_name)
                    self.data[col_name] = prep_NUMERIC.LabelEncoder(
                        self.data[col_name], col_name, self.artifacts_path, self.train
                    )
            elif key == "documentField":
                col_name = self.schema_param[key]
                self.sort_col_names.append(col_name)

                prep_text = prep_TEXT()

                self.data[col_name] = prep_text.get_process_text(
                    data=self.data[col_name],
                    col_name=col_name,
                    artifacts_path=self.artifacts_path,
                    Training=self.train,
                )

    def define_labels(self):
        labels = []
        for key in self.schema_param.keys():
            if "target" in key:
                labels.append(self.schema_param[key])

        if (
            len(labels) == 1
        ):  # If it's one labels then will return a string of that label only
            return labels[0]
        else:  # Otherwise it returns a list of labels
            return labels

    def drop_ids(self):
        self.data.drop(self.id_col, axis=1, inplace=True)

    def get_ids(self):
        return self.data[self.sort_col_names[0]]

    def sort_as_schem(self):
        """To ensure the consistancy of inputs are the same each time"""
        self.data = self.data[self.sort_col_names]

    def get_id_col_name(self):
        return self.id_col

    def save_label_pkl(self):
        """Saves labels as pickle file to call them laters and know the labels column later for invers encode"""
        if self.train:
            path = os.path.join(self.artifacts_path, "labels.txt")
            # Will save it in a txt file
            with open(path, "w") as f:
                if isinstance(self.LABELS, str):  # If it's one label not multiple
                    f.write(self.LABELS)
                else:
                    for label in self.LABELS:
                        f.write(label + "\n")

    def __split_x_y(self):
        self.y_data = self.data[self.LABELS]
        self.x_data = self.data.drop([self.LABELS], axis=1)
        return self.x_data, self.y_data

    def __train_test_split(self, train_ratio=0.8):
        self.__split_x_y()
        x_train_indx = int(train_ratio * len(self.x_data))
        self.x_train = self.x_data.iloc[:x_train_indx, :]

        if isinstance(self.LABELS, str):  # If it's one single label not multiple labels
            self.y_train = self.y_data.iloc[:x_train_indx]
            self.y_test = self.y_data.iloc[x_train_indx:]
        else:  # If it's multiple labels
            self.y_train = self.y_data.iloc[:x_train_indx, :]
            self.y_test = self.y_data.iloc[x_train_indx:, :]

        self.x_test = self.x_data.iloc[x_train_indx:, :]

        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_train_test_data(self):
        """returns:
        x_train, y_train, x_test, y_test
        """
        if self.gen_val_data:
            self.__train_test_split()
            return (
                self.x_train.to_numpy(),
                self.y_train.to_numpy().reshape((-1, 1)),
                self.x_test.to_numpy(),
                self.y_test.to_numpy().reshape((-1, 1)),
            )
        else:
            self.__split_x_y()
            return self.x_data.to_numpy(), self.y_data.to_numpy().reshape((-1, 1))

    def get_data(self):
        return self.data

    def invers_labels(self, data):
        """Handles only one label currently"""
        path = os.path.join(self.artifacts_path, "labels.txt")
        new_labels_list = []
        with open(path, "rb") as f:
            for line in f:
                new_labels_list.append(line)

        if len(new_labels_list) == 1:  # If it reads only one line which is one label
            labels = new_labels_list[0].decode().replace("\n", "")
        else:
            labels = new_labels_list.decode().replace("\n", "")

        inv_data = prep_NUMERIC.Inverse_Encoding(data, labels, self.artifacts_path)
        return inv_data


# ----------------------------------------------------------


class prep_TEXT:
    def __init__(self):
        """This class handles string features"""
        pass

    def get_process_text(
        self, data, col_name=None, artifacts_path=None, Training=False
    ):
        """Don't need it with tensorflow, textvectorizer layer will handle it"""
        if Training:
            self.adapt_text_vectorizer(
                data=data,
                col_name=col_name,
                artifacts_path=artifacts_path,
                Training=Training,
            )
        return data

    def adapt_text_vectorizer(
        self, data, col_name=None, artifacts_path=None, Training=False
    ):
        """It adapts the text on the first call, then exported as a model layer and transformation happens inside the model"""

        max_tokens = len(np.unique([text.split() for text in data]))
        max_length = round(sum([len(i.split()) for i in data]) / len(data))

        text_vectorizer = TextVectorization(
            max_tokens=max_tokens,  # how many words in the vocabulary (all of the different words in your text)
            standardize="lower_and_strip_punctuation",  # how to process text
            split="whitespace",  # how to split tokens
            output_mode="int",  # how to map tokens to numbers
            output_sequence_length=max_length,
        )

        text_vectorizer.adapt(data)
        self.save_text_vectorizer(
            text_vectorizer=text_vectorizer, artifacts_path=artifacts_path
        )
        return text_vectorizer

    def save_text_vectorizer(self, text_vectorizer, artifacts_path):
        path = os.path.join(artifacts_path, TEXT_VECTORIZER_NAME)
        pickle.dump(
            {
                "config": text_vectorizer.get_config(),
                "weights": text_vectorizer.get_weights(),
            },
            open(path, "wb"),
        )

    @classmethod
    def load_text_vectorizer(self, main_path=ARTIFACTS_PATH):
        path = os.path.join(main_path, TEXT_VECTORIZER_NAME)

        pickle_load_file = pickle.load(open(path, "rb"))
        text_vectorizer = TextVectorization.from_config(pickle_load_file["config"])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        text_vectorizer.set_weights(pickle_load_file["weights"])

        return text_vectorizer


# -----------------------------------------------------------


class prep_NUMERIC:
    """This class handles Numeric features"""

    def __init__(self):
        pass

    @classmethod
    def LabelEncoder(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name + ".pkl")
        if Training:
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data)
            pickle.dump(encoder, open(path, "wb"))
        else:
            encoder = pickle.load(open(path, "rb"))
            encoded_data = encoder.transform(data)
        return encoded_data

    @classmethod
    def get_Label_Encoder(self, col_name=None, artifacts_path=ARTIFACTS_PATH):
        if col_name is None:
            path = os.path.join(artifacts_path, "labels.txt")
            new_labels_list = []
            with open(path, "rb") as f:
                for line in f:
                    new_labels_list.append(line)

            if (
                len(new_labels_list) == 1
            ):  # If it reads only one line which is one label
                labels = new_labels_list[0].decode().replace("\n", "")
            else:
                labels = new_labels_list.decode().replace("\n", "")
        path = os.path.join(artifacts_path, labels + ".pkl")
        encoder = pickle.load(open(path, "rb"))
        return encoder

    @classmethod
    def Inverse_Encoding(self, data, col_name, artifacts_path):
        path = os.path.join(artifacts_path, col_name + ".pkl")
        encoder = pickle.load(open(path, "rb"))
        encoded_data = encoder.inverse_transform(data)
        return encoded_data

    @classmethod
    def handle_id(self, data):
        return data

    @classmethod
    def Min_Max_Scale(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name + ".pkl")
        if self.Training:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            pickle.dump(scaler, open(path, "wb"))
        else:
            scaler = pickle.load(open(path, "rb"))
            scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
        return scaled_data

    @classmethod
    def Standard_Scale(self, data, col_name, artifacts_path, Training=False):
        path = os.path.join(artifacts_path, col_name + ".pkl")
        if self.Training:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
            pickle.dump(scaler, open(path, "wb"))
        else:
            scaler = pickle.load(open(path, "rb"))
            scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
        return scaled_data
