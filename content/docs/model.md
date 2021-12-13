
---
weight: 1
title: Model
---

# Model



The following section will discuss how the pre-training and training models are constructed. 

To re-iterated, pre-training a Barlow head and then inheriting the head directly in the training loop led to poor results. While we saw that the shared head was converging very smoothly for the initial pre-training with Barlow Loss, and showed excellent results in pushing together the embeddings, it had very poor one-shot performance in Document Ranking, and did not converge more quickly. Consequently, we hypothesized that the model was not sufficiently complex to capture both the Barlow objective (push together question and conversation embeddings) and the Ranking objective (score highly on document ranking). That is, after training for the Barlow objective, the head was quickly overwritten when it was tasked with optimising for the Ranking objective. To counteract this (and thus get a true evaluation of applying Barlow Twins as a pretraining step), we decided to freeze the Barlow Twins head during training. Thus, the model learns to rank documents with a new head *without unlearning* the embeddings from the pretraining.


The two Models are demonstrated below, where <a style="color:tomato">red</a> indicates sub-models with frozen parameters and where <a style="color:dodgerblue">blue</a> indicates sub-models with trainable parameters.

> **Pre-training Model**
> 1. <p style="color:tomato"> BERT</p>
> 2. <p style="color:dodgerblue"> Barlow Head</p>

> **Training Model**
> 1. <p style="color:tomato"> Pre-training Model</p>
>    <p style="color:tomato">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Bert<p>
>    <p style="color:tomato">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Barlow Head</p>
> 2. <p style="color:dodgerblue"> Ranking Head</p>



## BERT Base
To delve further into the code, the BERT base simply passes the `input_id` and `attention_masks` from the Bert tokeniser through the Danish BERT model, downloaded from [Hugging Face](https://huggingface.co/Maltehb/danish-bert-botxo/blob/main/README.md). 

We made an initial choice to only train the head (for time and complexity reasons), therefore in `init` we freeze all BERT paramters so they will never be updated.

```python 
class BERT(nn.Module):
    def __init__(self, config):

        super(BERT, self).__init__()
        # Reference: https://huggingface.co/Maltehb/danish-bert-botxo/blob/main/README.md
        self.bert = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")

        # freeze all the parameters in BERT. This prevents updating of model weights during fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False


    # define the forward pass
    def forward(self, input_id, mask):

        # Bert
        BERT_output = self.bert(input_id, attention_mask=mask)

        return BERT_output
```

## Barlow Head

The goal of this model is to push together question and conversation encodings. 

We apply the following steps:
1. capture the CLS token from the BERT output.
2. Pass through a fully connected layer with batch normalisation
3. Apply a ReLU activation function
4. Apply Dropout
5. Pass through a second fully connected layer with batch normalisation

```python
class barlow_HEAD(nn.Module):

    def __init__(self, config):
        super(barlow_HEAD, self).__init__()
        # dropout layer
        self.dropout = nn.Dropout(config.dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 768)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(768, 768)
        # batch norm fc1
        self.bn1 = nn.BatchNorm1d(768)
        # batch norm fc2
        self.bn2 = nn.BatchNorm1d(768)

    def forward(self, BERT_output):

        # capture cls
        cls = BERT_output.last_hidden_state[:, 0, :]
        # FC 1 + batch norm
        x = self.bn1(self.fc1(cls))
        # relu activatiom
        x = self.relu(x)
        # dropout
        x = self.dropout(x)
        # FC 2 + batch norm
        x = self.bn2(self.fc2(x))

        return x
```

## Ranking Head

The goal of this model is to pair question and conversation encodings with document encodings to optimise the Document Ranking score.

We apply the following steps:
1. Pass through a fully connected layer with batch normalisation
2. Apply a ReLU activation function
3. Apply Dropout
4. Pass through a fully connected layer with batch normalisation

```python
class ranking_HEAD(nn.Module):

    def __init__(self, config):
        super(ranking_HEAD, self).__init__()
        # dropout layer
        self.dropout = nn.Dropout(config.dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768, 768)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(768, 768)
        # batch norm fc1
        self.bn1 = nn.BatchNorm1d(768)
        # batch norm fc2
        self.bn2 = nn.BatchNorm1d(768)

    def forward(self, x):

        # FC 1 + batch norm
        x = self.bn1(self.fc1(x))
        # relu activatiom
        x = self.relu(x)
        # dropout
        x = self.dropout(x)
        # FC 2 + batch norm
        x = self.bn2(self.fc2(x))

        return x

```


## Pre-training Model
The pre-training model simply passes the `input_id` and `mask` from the tokeniser through the bert model and the barlow head. The BERT parameters are frozen in the [Bert Base](#bert-base) `init`, so only the [Barlow Head](#barlow-head) parameters train. 
```python
class pretrainingModel(nn.Module):

    def __init__(self, config):
        super(pretrainingModel, self).__init__()
        self.bert = BERT(config)
        self.barlow_HEAD = barlow_HEAD(config)

    def forward(self, input_id, mask):
        BERT_output = self.bert(input_id, mask)
        x = self.barlow_HEAD(BERT_output)

        return x
```

## Training Model
In the Training-Model, the [pre-training model](pre-training-model) (including both [Bert Base](#bert-base) and [Barlow Head](#barlow-head)) is passed as a parameter. All model parameters are frozen, such that the training model will not update parameters from the pre-training model. During a forward pass, input is passed through the frozen pre-training model and then through an addition [Ranking Head](#ranking-head) which is trained with the objective of optimising document ranking scores.

Moreover, the training-model includes a method `set_pretrain_model_to_eval` that is called from the main script. When the Training Model is set to train mode, we want to ensure that only the ranking head is actually in train mode, while all frozen pre-training sub-models remain in eval mode. Thus, `set_pretrain_model_to_eval` sets the [pre-training model](pre-training-model) back to eval.

```python
class trainingModel(nn.Module):

    def __init__(self, pretrainingModel: pretrainingModel, config):
        super(trainingModel, self).__init__()

        # inherit the BERT and BarlowHEAD from pretraining model
        self.pretrainingModel = pretrainingModel
        # freeze all the parameters in barlow_HEAD. 
        # This embedding head shouldn't learn a new task. 
        # Ranking head needs to adapt to embedding provided.
        for param in self.pretrainingModel.parameters():
            param.requires_grad = False
            
        self.ranking_HEAD = ranking_HEAD(config)
    
    def set_pretrain_model_to_eval(self):
        self.pretrainingModel.eval()
    
    def forward(self, input_id, mask):
        x = self.pretrainingModel(input_id, mask)
        x = self.ranking_HEAD(x)

        return x
```