# BERT-CRF-NER
Using the very classicial network structure BERT-CRF to do named entity recognition task.


## train

```bash
pip install -r requirements.txt
python -m visdom.server
python run_bert_lstm_crf.py --file_path path_to_nerdata --model_name_or_path path_to_bert --save_dir path_to_checkpoints
```

**Note**

- **path_to_nerdata** refers to the folder where conll data is stored. There should be three files in this folder: train.txt,test.txt,valid.txt. For example: data/conll03/
- **model_name_or_path** refers to the folder where bert model is located, like xx/bert-base-uncased
- **save_dir** refers to the folder where checkpoints will be saved, like saved_model/


## test

```bash
python test.py
```

**Note**

- You need modify the variable save_dir and the path to conll data in test.py



## result

**The parameter configuration is saved in file args_dict where is under folder save_dir**

### Conll03 dev

| Model       | Precision | Recall | F1    |
| ----------- | --------- | ------ | ----- |
| BERT+Linear | 0.937     | 0.948  | 0.943 |
|  BERT+BiLSTM+CRF           |    0.933       |  0.945      |     0.939  |



### Conll03 test

| Model       | Precision | Recall | F1    |
| ----------- | --------- | ------ | ----- |
| BERT+Linear | 0.892     | 0.912  | 0.902 |
|    BERT+BiLSTM+CRF          |        0.914   |    0.927    |  0.920     |

