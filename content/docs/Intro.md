---
weight: 1
title: Intro
---

# Intro 

## About 
This website is a Handover Document by [Mikkel Odgaard](https://github.com/mikkelfo) and [Natasha Klingenbrunn](https://github.com/nklingen) for a collaboration project between Denmark's Technical University and [Raffle.ai](https://www.raffle.ai). The [final code](https://github.com/nklingen/BERT_Question_Answering) can be found on Github.

The website is broken down as follows:
1. The metrics for this project are outlined.
2. Next, the Timeline section will discuss the evolution of our project and discuss the rationale behind given design choices. Barlow Twins will be explained both conceptually and implementation-wise. 
3. The entire Model.py class will be broken down and explained in detail.
4. Finally, the discussion section will outline why believe our model was not successful, and touch upon our final thoughts.

## Motivation

Raffle.ai's BERT Document Ranking Model sees better performance on questions than on conversations. One hypothesis prior to beginning this project was that the BERT Ranking Model might project question and conversations into two seperate planes. Thus, while it may be very successful in one input-type, this does not translate to high performance for the other. The motivation behind this project is to introduce a pre-training step implementing Barlow Twins to push encodings together for similar questions and conversation.

Conceptually, Barlow twins aims to "collect" queries from the same answer id [think: high precision, but not necessarily accurate], while Document Ranking will put individual queries closer to the answer [think: high accuracy, but not necessarily precise]. Training the two loss functions one after the other will ideally encourage both accurate and precise results wherein one query type does not outperform the other.

![nat]({{< baseurl >}}/images/nat.png)


Secondly, we would like to interpret the results on performance. Having a successful implementation of the pre-training model will allow us to analyse the downstream effects in the BERT Ranking Model. We hope to learn whether having more similar embeddings for questions and conversations has a positive effect on the model, or whether the model's own learned representations are better. While we did not find it likely to have a better initial performance (if projecting all inputs from the same answer_id together had a lower loss in the downstream task, the original model would have done it anyways), but hoped that we could see other indicators of pre-training benefitting the model, such as a quicker rate of converging, for example. 