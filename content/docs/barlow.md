---
weight: 2
title: Barlow

mathjax: true
---

# Barlow

## Intro
Barlow Twins takes a batch of samples, applies noise to generate two distored versions, then passes both versions through two identical networks to get their corresponding embeddings. The Barlow loss is then computed on the embeddings, wherein the goal is to get the cross-correlation matrix between the embeddings as close as possible to the identity matrix. In this way, the embeddings of the two versions of the sample are encourraged to be similar, while redundancy between the components of the vectors is penalized.

*"Barlow Twins is competitive with state-of-the-art methods for self-supervised learning while being conceptually simpler, naturally avoiding trivial constant (i.e. collapsed) embeddings, and being robust to the training batch size."* From: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf)

In our project, we implement Barlow Twins to encourage similar embeddings between questions and conversations with the same `answer_id`. The assumption here is that questions and conversations that have the same answer must be related in some capacity, and moreover, that they must share an underlying concept. Hereby, the two "versions" can be likened to distortion of the same underlying concept.

## Theory

Whereas the original Barlow Paper takes a batch of samples and applies noise, in our implementation we already have the distorted matrices. For a given `answer_id`, this is simply a batch with associated questions (f), and a batch with associated conversations (g). In our dataloader, we ensure the batches are of equal size. 

![Barlow 1]({{< baseurl >}}/images/Barlow_1.png)

Now we compute the correlation between f and g. 
![Barlow EQ]({{< baseurl >}}/images/Barlow_eq.png)

This yields the following correlation matrix:
![Barlow Matrix]({{< baseurl >}}/images/Barlow_matrix.png)

Having now computed the correlation matrix, we want to encourage it to resemble the identity matrix. Hereby, we have two terms. In the `invariance_term` we encourage the diagonals (marked in grey) to be close to 1 and hereby for the model to be distortion agnostic, while in the `redundancy_reduction_term` we encourage all off-diagonals to be close to 0.


![Barlow 3]({{< baseurl >}}/images/Barlow_3.png)


## Implementation

```python

for batch in B:

    # clear previously calculated gradients
    optimizer.zero_grad()

    # push the batch to device
    batch = [r.to(device) for r in batch]
    input_encoding_ques, input_encoding_conv, _, _, attention_mask_conv, attention_mask_ques = batch

    barlow_sample_batch_size = input_encoding_ques.squeeze().shape[0]

    # pass in tokens from question to get BERT output
    f = model(
        input_encoding_ques.squeeze(), attention_mask_ques.squeeze()
    )  # batch_size (10) x hidden size (768)

    # pass in tokens from conv to get BERT output
    g = model(
        input_encoding_conv.squeeze(), attention_mask_conv.squeeze()
    )  # batch_size (10) x hidden size (768)

    # normalize along the batch dimensions, thus we have the normalized features across all batches
    f_norm = (f - f.mean(0)) / f.std(0)
    g_norm = (g - g.mean(0)) / g.std(0)

    # cross-correlation matrix
    c = torch.matmul(f_norm.T, g_norm)/ barlow_sample_batch_size

    invariance_term = torch.diagonal(c).add_(-1).pow_(2).sum()
    redundancy_reduction_term = off_diagonal(c).pow_(2).sum()
    loss = invariance_term + lambd * redundancy_reduction_term

```

Now to discuss our implementation. We create a specific dataloader for Barlow that passes through a batch of questions and a batch of conversations with the same `answer_id`, of equal size. Both batches are passed through the same model (a frozen BERT & a trainable barlow head)

The the cross-correlation matrix c is computed as discussed previously, and finally normalized with the corresponding batch size.

`c = (f - mean(f))*(g - mean(g))/(std(f) * std(g))/batch_size`, 

The reason for normalising the cross-correlation matrix is because the batch_size is not constant between new `answer_ids`. Some `answer_ids` have only a few questions and conversations, where as others may have hundreds. Moreover, the two classes may be imbalanced for any given `answer_id`. To solve this, and to maximize the batch sizes, the data_loader first computes the maximal possible batch size for the `answer_id`. 

`batch_size = min(len(questions), len(conversations))` 

Then it takes all datapoints from the smaller set, and samples from the larger set until we have an equal number of data points from both sets. 

Lastly, the `invariance_term` and `redundancy_term` are computed using two helper functions, `diagonal` and `off_diagonal` that return flattened versions of all elements in the diagonal or off-diagonal. The two terms are implemented exactly as described in [Theory](#theory)