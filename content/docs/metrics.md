---
weight: 5
title: Metrics
---

```python
# Computes top k accuracy
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

# Computes top k average precision score
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
* LOSS: we log both the validation and train loss, so that we can watch for signs of overfitting. 
* ACCURACY and MAP: When we say top *k* we measure a success criteria as each instance where  the correct answer id lies in the top *k* ranking. 

MAP penalises answers lower in the ranking while accuracy is blind to this.
We log these metrics with respect to questions only, conversations only, and both overall. We decided to log metrics seperately for the input types, as this project aims to increase the general performance of the model, meaning that one type should not be significantly worse than the other. Thus, it was important for us to track how our changes affected each individual type. Furthermore, we loG accuracy and MAP with k = 3, k = 10, as well as k = 1 for accuracy. 
