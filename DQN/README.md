# Deep Q Network


## Running
```bash
python main.py
```
### Visualizing
To 'render' (show) the next episode to screen, create a file in this directory called 'render'.  This will cause the gym to render the next episode.  After this the render file will be deleted.  For example:
```bash
touch render
```



## Benchmarks

lr = log10 of learning rate
e  = log10 of epsilon annealing steps
s  = sequence length per observation
Tq = Steps between Q-syncronization
Tt = Steps between weight updates
Ts = Steps after which success is evident


 - If Success=FAIL, DQN variant has not been observed to succeed at the task
 - These are by no means optimal settings, just ones that are observed to work (i.e. it is very possible that the network could learn faster with different parameters.)

### Normal DQN

_All benchmarking in this section was done with a dense 3-layer neural network, with 400 x 400 x n_actions nodes._

| Environment    | Success  | lr  | e   | s | Tq  | Tt | Ts    |
| -------------- | -------- | --- | ___ | _ | ___ | __ | _____ |
| CartPole-v1    | True     | -4  | 4.7 | 1 | 64  | 4  | 120 k |






### Citations
If you use this code, you should probably cite the following papers, depending on which features you use:

 - Original DQN paper:
   - https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
 - Prioritized buffer:
   - https://arxiv.org/abs/1511.05952
- Double Q learning
   - https://arxiv.org/abs/1509.06461
- Multi-step learning
   -  https://arxiv.org/abs/1710.02298

The original temporal difference paper (Sutton, 1988) is here: https://pdfs.semanticscholar.org/9c06/865e912788a6a51470724e087853d7269195.pdf


