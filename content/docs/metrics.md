---
weight: 2
title: Metrics
---

# Metrics 
> To compute top k accuracy
```python
def k_accuracy(scores, answer_ids, target_answer_ids, k):
    assert  k >= 1
    if  k == 1:
        prediction_indices = torch.argmax(scores, dim=1)
        prediction_answer_ids = answer_ids[prediction_indices]
        return (prediction_answer_ids == target_answer_ids)
    elif  k > 1:
        prediction_indices = torch.topk(scores, k).indices
        prediction_answer_ids = answer_ids[prediction_indices]
        return (prediction_answer_ids == target_answer_ids.unsqueeze(1)).any(1)
```
> To compute top k average precision score (from Raffle's codebase)
```python
def batch_map(relevance, k):
    top_k = relevance[:, :k]
    batch_size = top_k.size(0)
    pos = torch.arange(1, top_k.size(1) + 1, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1).to(top_k.device)
    csum, num_ans = top_k.cumsum(1), top_k.sum(1)
    apk = ((csum / pos * top_k).sum(1) / num_ans)
    apk = torch.masked_select(apk, ~torch.isnan(apk))
    return  apk.sum()
```

We log 3 types of metrics: loss, accuracy and average precision (MAP). 

<aside class="notice">
When we say top *k* we measure a success criteria as each instance where the correct answer id lies in the *k* top scores. Whereas accuracy is blind to how high the relative score was, as long as it still falls in the top k, MAP will still (slightly) penalize correct answers that were not ranked highly enough. 
</aside>


- Loss
    * Validation Loss
    * Train loss
- Accuracy
    * Questions
        * k = 1, k = 3, k = 10
    * Conversations
        * k = 1, k = 3, k = 10
    * Overall
        * k = 1, k = 3, k = 10
- MAP
    * Questions
        * k = 3, k = 10
    * Conversations
        * k = 3, k = 10
    * Overall
        * k = 3, k = 10

As our project aims to increase the general performance of the model, this means that we want to identify if one query type is significantly worse than the other. Thus, it was important for us to track how our changes to the model affected each of the query types. Moreover, we specifically looked at the accuracy top 3 metrics to compare with Raffle's own model. Lastly, we used the loss metrics to monitor for signs of overfitting. 




