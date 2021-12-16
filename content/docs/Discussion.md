---
weight: 6
title: Discussion

mathjax: true
---

# Discussion

<aside class="notice">
Nat
</aside>


We can see that Barlow Head accomplishes what it needs to do. (it works) but still that it breaks the downstream BERT task. We tried both frozen and unfrozen heads, and in both cases BERT is much worse off with the pretraining than with random initialized weights. We made sure it can't be a scaling issue as everything is normalized. 

Take-away, BERT makes its own representations in the encoding space that work well for document ranking, and tweaking these seems to hurt the model. Maybe moving questions and conversations together is an idea that makes sense from a human perspective, but interferes with the models deeper understanding of the datatypes. We don't deeply understand how the model represents the data.

Do queries with the same answer id even have the same question ?? sometimes yes sometimes no

is the CLS token sufficient ?

Why does the training step continue to reduce cluser sizes after pretraining ?? Does this indicate that the model *does* so something good ? 

<aside class="notice">
Mikkel
</aside>

Seeing as project did not yield any improvements and instead significantly worsen it, the question to ask is why. Why does BERT and Barlow work seperately, but completely fail when coupled. 

One of the reasons can be the assumption that questions and conversations with the same `answer id` should lie in the same latent space. This might be a very human intuition that does not necessarily translate into computer understanding. Moreover, our decision to only include the CLS token might also influence this. The CLS token is the classification token and BERT's understanding of classification might differ from humans. Even though a topic may be the same for a question and a conversation, the language in short and concise text (questions) might put it into a different classification than noisy and long text (conversations).

Secondly, we notice that the pretrained model is more or less broken for the ranking task. The model totally fails and has an accuracy slightly above 0%, much less than BERT achieves. This hints to the fact that there might be a fundamental problem, not just that Barlow might be a wrong assumption. Raffle.ai has had previous experience with BERT's apparent resistance to pretraining tasks. The underlying problem with pretraining might be that it interfers too much with the fundamental understandment of language by BERT. Additionally, there might be a need for special integration, rather than a trained head put ontop of BERT. However, these considerations requires much energy to investigate and we leave this for the future work. 
