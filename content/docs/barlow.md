---
weight: 2
title: Barlow

mathjax: true
---

# Barlow

The original paper, [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/pdf/2103.03230.pdf) proposes a loss function to encourage a model to create embeddings that are invariant to distortion. 

<aside class="success">
"Barlow Twins strongly benefits from the use of very high-dimensional embeddings" and "does not require large batches"
</aside>


Their architecture is as follows (simplified explanation): 
1. take an input image and apply distortions to it to
2. feed the distorted images through an encoder network and a projector network to get their respective embeddings
3. compute the Barlow Loss between the embeddings.

The Barlow Loss objective function compares the cross correlation between the embeddings of the distorted images with the identity matrix, essentially pushing them to be similar. 


$$L_{BT}\triangleq \sum_i(1-C_{ii})^2 + \lambda \sum_i \sum_{j \neq i}(C_{ij})^2$$


where the first term is the `invariance term` and the second term is the `redundancy reduction term`

The code from the [paper](https://arxiv.org/pdf/2103.03230.pdf) is as follows:

```python
# f: encoder network
# lambda: weight on the off-diagonal terms
# N: batch size
# D: dimensionality of the embeddings
#
# mm: matrix-matrix multiplication
# off_diagonal: off-diagonal elements of a matrix
# eye: identity matrix

for x in loader: # load a batch with N samples
    # two randomly augmented versions of x
    y_a, y_b = augment(x)
    # compute embeddings
    z_a = f(y_a) # NxD
    z_b = f(y_b) # NxD
    # normalize repr. along the batch dimension
    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
    z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
    # cross-correlation matrix
    c = mm(z_a_norm.T, z_b_norm) / N # DxD
# loss
c_diff = (c - eye(D)).pow(2) # DxD 
loss = c_diff.sum()
    # optimization step
    loss.backward()
    optimizer.step()
```