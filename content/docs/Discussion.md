---
weight: 6
title: Discussion

mathjax: true
---

# Discussion

## Barlow
Our main challange for this project was to implement Barlow on a new domain, and for a new task. The Barlow Loss was originally used to train a model to be robust to noise. However, we implemented it instead to push embeddings closed together in a latent space. Moreover, we see that this implementation was indeed successful as we confirmed from the converging loss, decreasing distance to cluster-centroid, and visual PCA interpretation. 

## Barlow with Document Ranking BERT

Moreover, we created a simplified, working model of Raffle's Document Ranking BERT. However, as soon as we coupled our barseline with the Barlow Pre-training, the model performed significantly more poorly than the original baseline. In fact, the starting accuracy was nearly 0, giving the same performance as randomly guessing. 

We tried with both frozen and unfrozen parameters being passed from the pretraining model to the training model, but in neither case could we see any improvement. We also hypothesized that this may be because the scaling is off, meaning that the embeddings are in an entirely different scale during pre-training than during training. This led us to implement batch-normalization in all the models to see if there was any improvement. However, the model did not improve.

When looking at the Barlow part of the results section, it looks quite peculiar. We see that Barlow trains nicely, but the model is completely destroyed when document ranking with respect to the pre-decided metrics. But then when we look at the PCA plots and the distance table, the results point to a success. Both the PCA plots show more promising embeddings without a linear seperation, and the distance table clearly shows that the queries are closer in the latent space, both to each other but also to their answer. This appears quite counterintuitive to our metric results.
 
We consider the possibility that the two models might just not be compatible. Essentially, maybe pushing the embeddings together is simply an inherently poor idea based on a faulty assumption that questions and conversations with the same `answer id` should lie in the same latent space. This might be a very human intuition that we are imposing on the model, which may not necessarily translate into computer understanding. 

We also notice that the pretrained model is more or less broken for the ranking task. This hints to the fact that there might be a problem in the integration between the two models. Raffle.ai has had previous experience with BERT's apparent resistance to pretraining tasks. In other words, there might be a need for a more careful integration, rather than simply passing forward the trained BARLOW head put ontop of BERT which might interfere with BERT's understanding of language. Given our time constraints, these considerations were beyond the scope of this project. We propose this for the future work.
