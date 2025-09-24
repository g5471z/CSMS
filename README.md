# CSMS
CSMS is our proposed class code summary model based on Transformer, which integrates method summary information to encode and decode. This work is completed based on [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum). We incorporated class code method summary sequences into the original work's code and added processing units for these sequences within the model. This enables parameter sharing within the encoder and joint decoding by the decoder.



## Before training

### (1) prerequisites

Before running the code, make sure you have the following prerequisites:
 * Java Development Kit (JDK)

### (2) get source code

 * source code
```
    git
```
The non-anonymous link to support repository cloning will be modified when time permits.

The process requires Linux and Python 3.6 or higher. It also requires installing PyTorch version 1.3 or higher. Its other dependencies are listed in requirements.txt. CUDA is strongly recommended for speed, but not necessary.

### (3) preprocess data
1)Prepare the dataset with _jsonl_ format.In the experiment, we used **ClassSum** dataset and **HRCE** dataset.The **ClassSum** dataset is downloaded from the website https://github.com/classsum/ClassSum .The **HRCE** dataset is downloaded from the website https://github.com/Haohaoi123/HRCE .

2)Run the `jsonl2text.py` file located at `data/java/read_json/`.You will get the dataset file in _txt_ format.Remember to modify the path of input data.

3)Run the `get_class_sum.py` file located at `data/java/get_method_summaries/`.You will get the summary of all the code in _txt_ format.We use [CodeT5](https://github.com/salesforce/CodeT5) to generate summary for each method code snippet.Remember to modify the path of input data.


## Training/Testing Models

To perform training and evaluation, first go the scripts directory associated with the target dataset.

```
$ cd  scripts/java
```

To train/evaluate a model, run:

```
$ bash script_name.sh GPU_ID MODEL_NAME
```

For example, to train/evaluate the transformer model, run:

```
$ bash transformer.sh 0,1 code2jdoc
```

#### Generated log files

While training and evaluating the models, a list of files are generated inside a `tmp` directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).
- **MODEL_NAME_beam.txt**
  - Log file for evaluation (beam).
- **MODEL_NAME_beam.json**
  - The predictions and gold references are dumped during evaluation (beam).

**[Structure of the JSON files]** Each line in a JSON file is a JSON object. An example is provided below.

```json 
{
    "id": 0,
    "code": "private int current Depth ( ) { try { Integer one Based = ( ( Integer ) DEPTH FIELD . get ( this ) ) ; return one Based - NUM ; } catch ( Illegal Access Exception e ) { throw new Assertion Error ( e ) ; } }",
    "predictions": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "references": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "bleu": 1,
    "rouge_l": 1
}
```

#### Generating Summaries for Source Codes

We may want to generate summaries for source codes using a trained model. And this can be done by running `generate.sh` located at `scripts/`. The input source code file must be under `java` directory.


```
$ cd scripts
$ bash generate.sh 0 code2jdoc input_file_name
```

The above command will generate `tmp/code2jdoc_beam.json` file that will contain the predicted summaries.

## Acknowledgement

We borrowed and modified code from [NeuralCodeSum](https://github.com/wasiahmad/NeuralCodeSum). We would like to expresse our gratitdue for the authors of this repository.

NeuralCodeSum: Uddin Ahmad W., Chakraborty S., Ray B.,et al., "A Transformer-based approach for source code summarization," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics,2020, pp.4998--5007. (https://github.com/wasiahmad/NeuralCodeSum)

