---
weight: 6
title: Discussion

mathjax: true
---

# Discussion

<aside class="notice">
Nat \ Mikkel
</aside>


We can see that Barlow Head accomplishes what it needs to do. (it works) but still that it breaks the downstream BERT task. We tried both frozen and unfrozen heads, and in both cases BERT is much worse off with the pretraining than with random initialized weights. We made sure it can't be a scaling issue as everything is normalized. 

Take-away, BERT makes its own representations in the encoding space that work well for document ranking, and tweaking these seems to hurt the model. Maybe moving questions and conversations together is an idea that makes sense from a human perspective, but interferes with the models deeper understanding of the datatypes. We don't deeply understand how the model represents the data.
