---
weight: 5
title: Timeline

---

# Timeline
The project went through 4 overall phases in regards to the construction of the model. 
1. Fundamental construction 
2. Exploratory Data Analysis (EDA)
3. Generic improvements
4. Barlow Twins

We will skip the majority of the details and focus on the key takeaways of each phase leading up to the final version of our model. This section will thus work as a brief summary of our work. 


## 1. Fundamental construction
Without going into much detail, the first part of the project was the implementation, and understandment, of BERT. The two key decisions in this phase was freezing BERT and only using the CLS tokens. The decision to freeze BERT parameters was primarily taken to increase speed, which is crucial in the early stages of development, with a small sacrifice in accuracy. Unfreezing BERT would have likely increased performance, but for the sake of consistency, we kept it frozen. The second decision was using only the CLS tokens. As one knows, the CLS token is the classification tokens. In such, the CLS token can be said to encapsulate all the information of the sentences, a summary or topic so to say. We deemed this token very important when ranking and was therefore our main, and only, source of information. However, as we will discuss later, this can also come with consequences. 

## 2.  Exploratory Data Analysis
The input data contains 4 main pieces of information. The title, the content, the type (questions or conversations) and answer id (the "true" label). Each piece of information should be analysed due to its importance. Grouping title and content together, in a group that can be called text, we end up with 3 types of information that must be analyzed. The answer ids can tell us something about the grouping of information, the question type can tell the balance of a dataset and the text can show whether questions and conversations are seperated. 

### 2.1 Answer ids
With some qualitative analysis, we investigated the assumption that input data which points to the same answer is about the same. This assumption is hard to fully investigate, but the crude investigated showed that the assumption is not completely off. However, to say that the assumption is true would be wrong. Based on this, we choose to apply the assumption, since it reduces the complexity of the problem and allows us to group the input data more easily. 

### 2.2 Dataset imbalance
The datasets (train and validation) are heavily skewed, having about 10 times the amount of conversations as opposed to questions. By including the generated questions, we get to a 2/3 split of conversations and questions, giving a more balanced dataset. Furthermore, we investigated casting short conversations as questions, as some conversations can be as short as a single sentence, esssentially making them a question. We chose to cast conversations with less than 100 characters into questions, making an almost perfectly balanced dataset. The balanced dataset is also crucial for the Barlow implementation later. 

### 2.3 PCA
In order to investigate whether questions and conversations, that has the same answer id, lie closely to eachother in latent space, we applied PCA to reduce the dimensionality down to 2D in order to visualize it. There was two main takeaways from the PCA plots. Firstly, there didn't appear to be distinct groupings based on answer ids and the embeddings was scattered in the latent space. There were certain areas with higher density of the same answer ids, but nothing that indicated seperation. Secondly, questions and conversations were not seperated and laid ontop of eachother. Howewer, conversatons were spread over the entire latent space, but questions appeared to be more clustered in certain areas. This might mean that BERT somewhat distinguishes between conversations and questions at a certain level.

## 3. Generic improvements
### 3.1 Preprocessing

```python
def TFIDF_standard(data):
    # capture the title and content
    title = [x["title"] for x in data]
    content = [x["content"] for x in data]

    # split by words
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    lookup = vectorizer.fit(content).vocabulary_

    input_string = []
    word_split = re.compile(r'(?u)\b\w+\b')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for i, document in enumerate(content):
        # split document into sentencces
        sentences = tokenizer.tokenize(document)
        sentence_scores = []
        for sentence in sentences:
            score = 0
            words = word_split.findall(sentence.lower())
            # split sentence into words and compute the total tf-idf score per sentence
            for word in words:
                score += lookup[word]

            # Relative score per word (incase of weird sentences (double periods) we have the if statement)
            if len(words) > 0: 
                score /= len(words)
            sentence_scores.append(score)

        # Relative score per character for each sentence
        sentence_scores = torch.tensor(sentence_scores)
        # Ranking of which sentence had the most important characters per length.
        sorted_indices = torch.argsort(sentence_scores)

        current_length = len(word_split.findall(title[i]))
        kept_indices = []

        # Greedy Knapsack solution to add sentences per average char weight
        for index in sorted_indices:
            word_count = word_split.findall(sentences[index])
            if current_length < TFIDF_answer_length:
                kept_indices.append(index)
                current_length += len(word_count)

        # rearrange the sentences from (most -> least important) to original order
        final_string = title[i] + "."
        for index in sorted(kept_indices):
            final_string += " " + sentences[index]
        input_string.append(final_string)
        
    return  input_string
```

As the input has to be of limited size, mainly due to speed and to reduce noise, we implement preprocessing to the input, so that we feed BERT with the, hopefully, most useful information. In order to do so, we applied TF-IDF in order to include the sentences with the most importance. Each sentence is scored by the relative score per word, which means summing each word's TF-IDF score and dividing by the number of words for the sentence. We then take the most important sentences, up to 128 tokens, and feed this to the model. 

A potential improvement was to utilize the sections.json, so that there was an equal amount of sentences per section, however this implementation did not yield any significant improvements. Moreover, increasing the amount of tokens, from 128 up to 258 and 512, also did not yield any significant improvements. Instead it merely increased runtime by a significant factor (Twice the amount of data takes about twice the amount of time). 

### 3.2 Dual encoder

```python
def  train(model, model_answers, optimizer, criterion, data_loader, answer_loader, cls_dictionary):
    answer_tokens = torch.stack(list(cls_dictionary.values())).to(device)
    answer_ids = torch.tensor(list(cls_dictionary.keys())).to(device)
    
    # Metrics
    total_loss = 0
    
    model.train()
    model_answers.train()

    answer_iterator = iter(answer_loader)

    for  step, (data_batch) in  enumerate(data_loader):
        # Keep updating the stale index (multiple updates for each ID each training step)
        answer_batch = next(answer_iterator)

        # clear previously calculated gradients
        optimizer.zero_grad()

        ''' Answer batch '''
        answer_encoding, _, attention_mask, batch_ids = answer_batch
        new_answer_tokens  = model_answers(answer_encoding, attention_mask)

        # Only update the relevant answer_tokens
        answer_tokens[(answer_ids == batch_ids.unsqueeze(1)).nonzero()[:,1]] = new_answer_tokens

        ''' Data batch '''
        input_encoding, _, attention_mask, target_answer_ids, question_type = data_batch
        question_tokens = model(input_encoding, attention_mask)

        # Calculate scores (batch_tokens vs stale_index_tokens)
        scores = torch.matmul(question_tokens, answer_tokens.T)
        target_answer_indices = (answer_ids == target_answer_ids.unsqueeze(1)).nonzero()[:, 1]

        ''' Compute loss '''
        loss = criterion(scores, target_answer_indices)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        # Important to detach from graph
        answer_tokens.detach_()
        
        total_loss += loss.item()
```

A significant improvement was seen when implementing a dual encoder setup. The setup uses two different heads, one for the input data and one for the answers. During a training step the model will update part of the stale index and this updated stale index is used to calculate the scores of the input data. In such, we collect gradients from both the indexing of answers and the scoring of questions and conversations. There is still only calculate a single loss based on the scoring, but this loss is backpropagated into two seperate heads now. 

This implementation had a more gradual learning curve, starting off worse than the other models, but was also to train for longer (10-20 epochs before stagnating) and giving significant improvements on all metrics. Most noticably, the dual encoder implementation surpasses the other models before the stagnate. This means that the dual encoders increased performance is not just due to the longer training. Stopping the dual encoder model at the same point as the others would have also yielded significant improvements. 


### 3.3 Gradient clipping, L2 regularization & batch normalization
We apply some of the common deep learning improvements, such as gradient clipping, l2 regularization and batch normalization. Without much investigation in these, we saw immediate improvements. These techniques help us stabilize the learning and prevent overfitting. 


## 4. Barlow Twins

