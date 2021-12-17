
---
weight: 5
title: Results

mathjax: true
---

# Results
Each model is set to terminate after 10 epochs without improvement of the validation loss. This is what causes the different termination times in the below plots. 

### Document Ranking Results
We begin with the baseline model, otherwise known as BERT with one head. We compare it with plots from our pre-processing experiments: TF-IDF pre-processing by section, or with 512 tokens. These results show us that the optimal pre-processing was just the initial implementation. Next, we implemented the Dual Encoder model, which we see gave a significant performance boost. We compare the three models via validation loss, Accuracy, and MAP.

> The best model is by far the dual encoder implementation. This implementation had a more gradual learning curve, starting off worse than the other single-head models, but was also to train for longer (10-20 epochs before stagnating) and giving significant improvements on all metrics. Most noticably, the dual encoder implementation surpasses the other models before they stagnate. This means that the dual encoder's increased performance is not just due to the longer training. Stopping the dual encoder model at the same point as the others would have also yielded significant improvements. 

![Validation loss of the generic improvements]({{< baseurl >}}/images/validation_loss.png)

| Model / Top 3 accuracy| Total | Question | Conversation|
| -- | -- | --|--|--|
| Baseline | 0.44 | 0.32 | 0.50
| Sections | 0.43 | 0.29 | 0.49
| Dual encoder | 0.49 | 0.34 | 0.57
| Baseline 512 tokens | 0.43 | 0.32 | 0.49

> We see that the model is significantly better at ranking conversations than questions, which is quite unexpected. Conversations are usually more noisy and harder to interpret than questions. However, conversations also contain additional information that the model may have benefitted from. 


| Model / Top 3 MAP| Total | Question | Conversation|
| -- | -- | --|--|--|
| Baseline | 0.36 | 0.24 | 0.42
| Sections | 0.33 | 0.23 | 0.41
| Dual encoder | 0.40 | 0.26 | 0.47
| Baseline 512 tokens | 0.34 | 0.23 | 0.40


### Barlow Results

> There is a clear and smooth loss curve. This is particularly promising as, due to the sampling in each new batch, the model will rarely encounter the same data set twice (particularly if there is a large imbalance between the classes). 

![The loss during Barlow training]({{< baseurl >}}/images/barlow_loss.png)

### Barlow and Document Ranking

We will now compare the three models.
1. *baseline* - the Baseline Dual Encoder
2. *zero-shot* - the Barlow pre-training Model zero-shot on the training data
3. *pre-trained baseline* - the Barlow pre-training model with frozen parameters followed by the Dual Encoder. 

The pretraining's goal was to push questions and conversations together. In order to evaluate this, we will discuss two metrics. 
- First, we will qualitatively evaluate the differences in our PCA plots. 
- Secondly we will present a table of different distances mesaurements that will allow us to quantify the differences between the models. 

The PCA plots contain the question and conversation centroids for every answer id. Each PCA plot consists of 3 subplots, where each plot varies in the method of PCA. 
- *High dim* calculates the centroids in the high dimension embeddings before projecting down to 2D. 
- *Med dim* reduces dimensionality to 30 before calculating the centroids then projects down to 2D. 
- *low dim* reduces to 2 dimensions before calculating the centroids. This is done to reduce the influence of the dimensionality reduction so that proper conclusions can be drawn. 

> PCA for Baseline

> Notice the clear seperation below of questions (1) and conversations (0).

![PCA for baseline]({{< baseurl >}}/images/centroids_baseline.png)


> PCA for Zeroshot

> While the zeroshot fails document ranking, there is a clear improvement in encouraging closer embeddings in plot. The clear seperation has been removed, although we still see some signs of seperation. 

![PCA for zeroshot]({{< baseurl >}}/images/centroids_zeroshot.png)

> PCA for Pre-trained Baseline

> We see that the pretrained model is slightly more seperated when compared to the zeroshot, but is still better than the baseline. It appears that the model still perserves some of its pretraining after training. 

![PCA for pretrained baseline]({{< baseurl >}}/images/centroids_pretrained.png)

From the above plots we clearly see that Barlow (and pretraining) accomplish the goals we want when looking at PCA-based metrics. However, we are more interested in the specific `answer id` groupings. Therefore, we create a table with the following two measurements to confirm the results above:
- a centroid distance, that is the average distance between the question and conversation centroid with the same `answer id`
- the average distance from every query to its corresponding answer. 

Both of these distances are calculated with the embeddings normalized in a range from 0 to 1.  

| Model 	| Avg. dist centroid | Avg. dist to answer| 
|--|--| --
| Baseline 	| 2.09	| 0.059
| Zero-shot 	| 1.6	| 0.058
| Pretrained Baseline | 1.07 | 0.032

<aside class="notice">
The average distance centroid is a distance metric in multiple dimensions, which is why it may be above 1 even though the embeddings themselves have been normalized.
</aside>

In comparison to the baseline's average distance to centroid:
- we expected the zero-shot model to have a **significantly lower average distance to centroid** since the Barlow Head is purely tasked with generating similar encodings for queries with the same answer id.
- we expected the pre-trained model to have a **lower average distance to centroid**, given that potentially it might spread out some of the encodings in the training head to perform better on Document Ranking.

From the PCA plots, the zero-shot model and the pretrained baseline that was allowed to train a new head for Document Ranking performed similarly. But in this table we see a clear descending progression in the average distance between centroids. There is about a 0.5 drop between each model. Furthermore, and the pretrained baseline is at approximately a 50% reduction from the baseline. This is counterintutitive because it seems that the model continued to push the embeddings together, even after the initial Barlow head was frozen and the new head was tasked purely with Document Ranking. 

In comparison to the baseline's average distance to answer:
- we expected the zero-shot model to have a  **slightly higher average distance to answer**. While it never directly trains to push the embeddings closer to the answer_id, we expect the groups to move together to middle-cluster that would be closer to the answer on average.
- we expected the pre-trained model to have a **similar average distance to answer**. Since it have the opportunity to train on the answers, we expected it to move closer to the answer than the zero-shot. 

We very oddly find that the average distance to answer is nearly the same as the baseline, without ever having seen the true answer. Moreover, the pretrained baseline also has a 50% reduction compared to the baseline. Overall, it looks like Barlow accomplishes its task of pushing both questions and conversation together, which in turn makes these closer to the answer they point to. In the discussion we will interpret this further. 

From this we would expect that the zero-shot would have similar performance to the baseline, and that the pre-trained baseline would outperform the baselne. However, this was very far from the results we saw.

> We see that the pretrained baseline has completely collapsed at the start of training. The pretrained baseline, and the zero-shot at step 0, is totally unequipped for the document ranking task.  Essentially, on running different variations of the model, the starting point is  very poor and the model does not converge more quickly.  While we only depicted top3 accuracy in the plot, the rest of the metrics look equally poor.

![Baseline vs. pretrained baseline]({{< baseurl >}}/images/pretraining.png)
 
![Train Loss]({{< baseurl >}}/images/TrainLoss.png)

![Validation Loss]({{< baseurl >}}/images/ValidationLoss.png)