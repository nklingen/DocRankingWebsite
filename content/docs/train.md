---
weight: 6
title: Train
---

# Train

We also want to go into more detail in the training loop


```python
def  train(model, model_answers, optimizer, criterion, data_loader, answer_loader, cls_dictionary):
    answer_tokens = torch.stack(list(cls_dictionary.values())).to(device)
    answer_ids = torch.tensor(list(cls_dictionary.keys())).to(device)
    
    # Metrics
    total_loss = 0
    
    model.train()
    model_answers.train()

    answer_iterator = iter(answer_loader)

    for  step, (data_batch) in  enumerate(data_loader):
        # Keep updating the stale index (multiple updates for each ID each training step)
        answer_batch = next(answer_iterator)

        # clear previously calculated gradients
        optimizer.zero_grad()

        ''' Answer batch '''
        answer_encoding, _, attention_mask, batch_ids = answer_batch
        new_answer_tokens  = model_answers(answer_encoding, attention_mask)

        # Only update the relevant answer_tokens
        answer_tokens[(answer_ids == batch_ids.unsqueeze(1)).nonzero()[:,1]] = new_answer_tokens

        ''' Data batch '''
        input_encoding, _, attention_mask, target_answer_ids, question_type = data_batch
        question_tokens = model(input_encoding, attention_mask)

        # Calculate scores (batch_tokens vs stale_index_tokens)
        scores = torch.matmul(question_tokens, answer_tokens.T)
        target_answer_indices = (answer_ids == target_answer_ids.unsqueeze(1)).nonzero()[:, 1]

        ''' Compute loss '''
        loss = criterion(scores, target_answer_indices)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        
        optimizer.step()
        
        # Important to detach from graph
        answer_tokens.detach_()
        
        total_loss += loss.item()
```