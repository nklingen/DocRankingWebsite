---
weight: 4
title: Timeline

---

# Timeline
This section will be a general summary of our work and the evolution of the project. The project went through 4 overall phases in regards to the construction of the model. 

2. Exploratory Data Analysis (EDA)
3. Document Ranking Model
4. Barlow Twins

## Exploratory Data Analysis
The input data contains 4 main pieces of information. The title, the content, the type (questions or conversations) and answer id (the "true" label). Each piece of information should be analysed due to its importance. Grouping title and content together, in a group that can be called text, we end up with 3 types of information that must be analyzed. The answer ids can tell us something about the grouping of information, the question type can tell the balance of a dataset and the text can show whether questions and conversations are seperated. 

### Answer ids
With some qualitative analysis, we investigated the assumption that input data which points to the same answer is about the same. This assumption is hard to fully investigate, but the crude investigated showed that the assumption is not completely off. However, to say that the assumption is true would be wrong. Based on this, we choose to apply the assumption, since it reduces the complexity of the problem and allows us to group the input data more easily. 

### Dataset imbalance
The datasets (train and validation) are heavily skewed, having about 10 times the amount of conversations as opposed to questions. By including the generated questions, we get to a 2/3 split of conversations and questions, giving a more balanced dataset. Furthermore, we investigated casting short conversations as questions, as some conversations can be as short as a single sentence, esssentially making them a question. We chose to cast conversations with less than 100 characters into questions, making an almost perfectly balanced dataset. The balanced dataset is also crucial for the Barlow implementation later. 

### PCA
In order to investigate whether questions and conversations, that has the same answer id, lie closely to eachother in latent space, we applied PCA to reduce the dimensionality down to 2D in order to visualize it. There was two main takeaways from the PCA plots. Firstly, there didn't appear to be distinct groupings based on answer ids and the embeddings was scattered in the latent space. There were certain areas with higher density of the same answer ids, but nothing that indicated seperation. Secondly, questions and conversations were not seperated and laid ontop of eachother. Howewer, conversatons were spread over the entire latent space, but questions appeared to be more clustered in certain areas. This might mean that BERT somewhat distinguishes between conversations and questions at a certain level.

## Document Ranking Model

The first part of the project was to implement and understand a BERT model trained for Document Ranking. 

### Preprocessing for BERT

> Step 1. Compute the TF-IDF score for each word in the corpus
```python
def TFIDF_standard(data):
	title = [x["title"] for x in data]
	content = [x["content"] for x in data]
	vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
	lookup = vectorizer.fit(content).vocabulary_
    input_string, kept_indices = []
```
> Step 2. Compute the relative TF-IDF score for each sentence
```python
	word_split = re.compile(r'(?u)\b\w+\b')
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	for i, document in enumerate(content):
		sentence_scores = []
		for sentence in tokenizer.tokenize(document):
			score = 0
			for word in word_split.findall(sentence.lower())
				score += lookup[word]
			num_words = max(1, len(words))
			sentence_scores.append(score/num_words) 
```
> Step 3. Order each sentence from most to least important based on relative TF-IDF score
```python
		sentence_scores = torch.tensor(sentence_scores)
		sorted_indices = torch.argsort(sentence_scores)
```
> Step 4. Take the max number of sentence (from most important to least) you can fit in 128 tokens
```python
        current_length = len(word_split.findall(title[i])) # title is always returned
		# Greedy Knapsack solution to add sentences per average char weight
		for index in sorted_indices:
			word_count = word_split.findall(sentences[index])
			if current_length < TFIDF_answer_length:
				kept_indices.append(index)
				current_length += len(word_count)
``` 
> Step 5. Re-order the sentences to the original order
```python
		final_string = title[i] + "."
		for index in sorted(kept_indices):
			final_string += " " + sentences[index]
		input_string.append(final_string)
		
	return  input_string
```

As the input to BERT has to be of limited size due to time and memory constraints, we preprocess the input, to remove noise in the data and feed BERT with only the most useful information. 

To do so, We decided to include the sentences with the highest importance according to relative sentence TF-IDF score. We computed this in the following way.


1. Compute the TF-IDF score for each word in the corpus
2. Compute the relative TF-IDF score for each sentence
    * First sum up the total TF-IDF score for each word in the sentence
    * Divide by the amount of words in the sentence to compute a relative score
3. Order each sentence from most to least important based on relative TF-IDF score
4. Take the max number of sentence (from most important to least) you can fit in 128 tokens
5. Re-order the sentences to the original order

The entire code is demonstrated on the right.

Additionally, we tried utilizing the sections.json file, and instead return an equal number of top TF-IDF scored sentences *per section*, however this implementation did not yield any significant improvements. 

Moreover, we also attempted to increase the amount of tokens from 128 to 258 and 512; this did not yield any significant improvements. Instead it merely increased runtime lineraly in proportion to the number of tokens. (Twice the amount of data took roughly twice the amount of time).

### BERT with one head

For clarity, question and conversations embeddings will be called "queries" in this section.

We will now discuss some design choices in implementing BERT. Namely, the two key decisions in this phase were **freezing BERT** and **only using the CLS tokens**. 
* The decision to freeze BERT parameters was primarily taken to increase speed and avoid running into issues of memory constraint, which was crutial for us given this was a short-term research project and we had GPU resource constraints. We knew this would entail a decrease in accuracy. Unfreezing BERT and adapting the model for all experiments would have likely increased performance. 
* The second decision was using only the CLS tokens. Briefly, the CLS token is first column of the BERT output, typically used for classification tasks. The CLS token can be said to encapsulate all the information of the input, to give a sort of general summary. We deemed this token most important for ranking. Thus, given the same speed and memory constraints, we chose to only use the CLS token and discard the rest of the BERT output. However, as we will discuss later, this can also come with consequences. 

The model was implemented as follows 

1. Compute the answer Database
We passed the answers through a BERT model with a head to get a database of answer encodings. 
2. Compute the query embeddings
Next, we passed the questions and conversations through the same BERT model with the same head to get their encodings. 
3. Document Ranking
    * To compute most related documents for a given query, we compute the cross product between the query embeddings and the entire answer database. 
    * We then find the argmax for each query, that is the answer document that had the highest score for that query.
    * We return the index of the argmax.

<aside class="notice">
This is a simplified explanation for the instance where k=1. To compute the top k most relevant documents, the indices of the top k argmax elements are returned for each query.
</aside>

### BERT with Dual Encoder

A significant improvement was seen when implementing a dual encoder setup. The setup uses two different heads, one for the input data and one for the answers. During a training step the model will update part of the stale index. This updated stale index is used to calculate the scores of the input data. In such, we collect gradients from both the indexing of answers and the scoring of questions and conversations. There is still only calculate a single loss based on the scoring, but this loss is backpropagated into two seperate heads now. 

<aside class="notice">
Move to results ??
</aside>

This implementation had a more gradual learning curve, starting off worse than the other models, but was also to train for longer (10-20 epochs before stagnating) and giving significant improvements on all metrics. Most noticably, the dual encoder implementation surpasses the other models before they stagnate. This means that the dual encoders increased performance is not just due to the longer training. Stopping the dual encoder model at the same point as the others would have also yielded significant improvements. 


### Gradient clipping, L2 regularization & batch normalization
We apply some of the common deep learning improvements, such as gradient clipping, and l2 regularization, and saw immediate improvements. These techniques help us stabilize the learning and prevent overfitting. 

Later, we also exchanged the gradient clipping after the model terminates with batch normalization in between the blocks of the model. 

### Hyperparameter Tuning

In order to fix a model for our expeirments we found a set of parameters using hyperparameter tuning. We tune with respect to the validation loss over 4 parameters: dropout rate, learning rate, batch size and weight decay. We utilized 'Weights & Biases' automatic hyperparameter tuning (sweeps) and chose to us Bayes optimisation to optimise the search. We do the sweep on the *baseline* model described above.

![Overview of all the runs]({{< baseurl >}}/images/sweep.jpg)

We fix the `max_length=128`, i.e. the amount of tokens fed to BERT, and the max number of `epochs=50`. The sweep finds the following parameters:

`batch size=50`
`dropout=0.11`
`learning rate=1.4e-4`
`weight decay=9.3e-4`

Moreover, 'Weights & Biases' tells us the importance and correlation of each parameter. It clearly shows the learning rate being the most influential, which is expected due to its large impact on training.

![Parameter importance]({{< baseurl >}}/images/param_importance.png)

## Barlow Twins

At this stage, we had a very basic, working Document Ranking Model. We now wanted to try out our hypothesis, that pushing together similar question and conversation encodings might improve the model.

### Theory
Barlow Twins takes a batch of samples, applies noise to generate two distored versions, then passes both versions through two identical networks to get their corresponding embeddings. The Barlow loss is then computed on the embeddings, wherein the goal is to get the cross-correlation matrix between the embeddings as close as possible to the identity matrix. In this way, the embeddings of the two versions of the sample are encourraged to be similar, while redundancy between the components of the vectors is penalized.

*"Barlow Twins is competitive with state-of-the-art methods for self-supervised learning while being conceptually simpler, naturally avoiding trivial constant (i.e. collapsed) embeddings, and being robust to the training batch size."* From: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)

In our project, we implement Barlow Twins to encourage similar embeddings between questions and conversations with the same `answer_id`. The assumption here is that questions and conversations that have the same answer must be related in some capacity, and moreover, that they must share an underlying concept. Hereby, the two "versions" can be likened to distortion of the same underlying concept.


Whereas the original Barlow Paper takes a batch of samples and applies noise, in our implementation we already have the distorted matrices. For a given `answer_id`, this is simply a batch with associated questions (f), and a batch with associated conversations (g). In our dataloader, we ensure the batches are of equal size. 

![Barlow 1]({{< baseurl >}}/images/Barlow_1.png)

Now we compute the correlation between f and g. 
![Barlow EQ]({{< baseurl >}}/images/Barlow_eq.png)

This yields the following correlation matrix:
![Barlow Matrix]({{< baseurl >}}/images/Barlow_matrix.png)

Having now computed the correlation matrix, we want to encourage it to resemble the identity matrix. Hereby, we have two terms. In the `invariance_term` we encourage the diagonals (marked in grey) to be close to 1 and hereby for the model to be distortion agnostic, while in the `redundancy_reduction_term` we encourage all off-diagonals to be close to 0.


![Barlow 3]({{< baseurl >}}/images/Barlow_3.png)


### Implementation

```python

for batch in B:
    ... 
    input_encoding_ques, input_encoding_conv, _, _, attention_mask_conv, attention_mask_ques = batch
    barlow_sample_batch_size = input_encoding_ques.squeeze().shape[0]

    # question (f) and conversation (g) encodings
    f = model(input_encoding_ques.squeeze(), attention_mask_ques.squeeze()) 
    g = model(input_encoding_conv.squeeze(), attention_mask_conv.squeeze())  

    # normalize along the batch dimensions, thus we have the normalized features across all batches
    f_norm = (f - f.mean(0)) / f.std(0)
    g_norm = (g - g.mean(0)) / g.std(0)

    # cross-correlation matrix
    c = torch.matmul(f_norm.T, g_norm)/ barlow_sample_batch_size

    # Barlow Loss
    invariance_term = torch.diagonal(c).add_(-1).pow_(2).sum()
    redundancy_reduction_term = off_diagonal(c).pow_(2).sum()
    loss = invariance_term + lambd * redundancy_reduction_term
    ...
```

Now to discuss our implementation. We create a specific dataloader for Barlow that passes through a batch of questions and a batch of conversations with the same `answer_id`, of equal size. Both batches are passed through the same model (a frozen BERT & a trainable barlow head)

The the cross-correlation matrix c is computed as discussed previously, and finally normalized with the corresponding batch size.

`c = (f - mean(f))*(g - mean(g))/(std(f) * std(g))/batch_size`, 

The reason for normalising the cross-correlation matrix is because the batch_size is not constant between new `answer_ids`. Some `answer_ids` have only a few questions and conversations, where as others may have hundreds. Moreover, the two classes may be imbalanced for any given `answer_id`. To solve this, and to maximize the batch sizes, the data_loader first computes the maximal possible batch size for the `answer_id`. 

`batch_size = min(len(questions), len(conversations))` 

Then it takes all datapoints from the smaller set, and samples from the larger set until we have an equal number of data points from both sets. 

Lastly, the `invariance_term` and `redundancy_term` are computed using two helper functions, `diagonal` and `off_diagonal` that return flattened versions of all elements in the diagonal or off-diagonal. The two terms are implemented exactly as described in [Theory](#theory)

