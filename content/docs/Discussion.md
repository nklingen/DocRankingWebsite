---
weight: 6
title: Discussion

mathjax: true
---

# Discussion

The main takeaway was that we could see that both the Document Ranking BERT and the Barlow models worked seperately. However, as soon as we coupled them, the model performed significantly worse than the original baseline. In fact, the starting accuracy was nearly 0, giving the same performance as randomly guessing. 

We tried with both frozen and unfrozen parameters being passed from the pretraining model to the training model, but in neither case could we see any improvement. We also hypothesized that this may be because the scaling is off, meaning that the embeddings are in an entirely different scale during pre-training than during training. This led us to implement batch-normalization in all the models to see if there was any improvement. However, the model did not improve.
 
Firstly, we consider the possibility that the two models might just not be compatible. Essentially, maybe pushing the embeddings together is simply an inherently poor idea based on a faulty assumption that questions and conversations with the same `answer id` should lie in the same latent space. This might be a very human intuition that we are imposing on the model, which may not necessarily translate into computer understanding. In fact, we do not even know how closely the two concepts of distance and relative ranking are linked. Potentially there is no correlation at all, and pushing the embeddings together is just an additional "hurdle" for the model to overcome. We propose that this be studied in Future Work, as an indicator for whether this project should be continued.

We notice that the pretrained model is more or less broken for the ranking task. This hints to the fact that there might be a fundamental problem in the integration between the two models. Raffle.ai has had previous experience with BERT's apparent resistance to pretraining tasks. In other words, there might be a need for a more careful integration, rather than simply passing forward the trained BARLOW head put ontop of BERT. Given our time constraints, these considerations were beyond the scope of this project. We propose this for the future work.