---
weight: 1
title: Intro
---

# Intro 

## About 
This website is a Handover Document by [Mikkel Odgaard](https://github.com/mikkelfo) and [Natasha Klingenbrunn](https://github.com/nklingen) for a collaboration project between Denmark's Technical University and [Raffle.ai](https://www.raffle.ai). The [final code](https://github.com/nklingen/BERT_Question_Answering) can be found on Github.

The website is broken down as follows:
1. A brief conceptual background is given for Barlow Twins and BERT.
2. Next, the Timeline section will discuss the evolution of our project and discuss the rationale behind given design choices. 
3. The code break-down for our model and barlow loss can be found in their respective sections.
4. Finally, the discussion section will outline why believe our model was not successful, and touch upon our final thoughts.

## Motivation

Raffle.ai's BERT Document Ranking Model sees better performance on questions than on conversations. The motivation behind this project is to leverage SSL to yield a more constant performance across input types. To accomplish this, we will introduce a pre-training step implementing Barlow Twins to push encodings together for similar questions and conversations. Having a successful implementation of the pre-training model will allow us to analyse the downstream effects in the BERT Ranking Model. 

For example, one hypothesis prior to beginning this project was that the BERT Ranking Model might project question and conversations into two seperate planes. Thus, while it may be very successful in one input-type, this does not translate to high performance for the other. We hope to learn whether having more similar embeddings for questions and conversations has a positive effect on the model, or whether the model's own learned representations are better. While we didn't find it likely to have a better initial performance (if projecting all inputs from the same answer_id together had a lower loss in the downstream task, it would have done it anyways), but hoped that we could see other indicators of pre-training benefitting the model, such as a quicker rate of converging, for example. 