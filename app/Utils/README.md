# Navigate Code

## 1- preprocess folder

consist of:

#### 1- schema_hanlder.py:

It reads the schema parameter using recursive function.

#### 2- preprocess.py:

Loads the schema handler and data, iterates over all columns to define which preprocess function to do. In case it's first time to preprocess data, the preprocess module saves artifiacts used during preprocessing to load it later when inference/prediction on new data is required.

prep_TEXT class, and prep_NUMERIC is where the processing happens. preprocess_data class only control, call, and defines if is training or inference to load or generate artifacts.

## 2- model_builder.py

Builds and train the model according to the passed data and model config json file. also saves and loads the model.

## 3- predictions_handler.py

Consist of predictor class where we pass the data to it. calls the preprocess to process data and pass it to loaded model, then produce results in labels as a pandas data frame.

## 4- utils.py

Consist of helping functions such as load_json_files.
