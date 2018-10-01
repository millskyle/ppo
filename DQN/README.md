# Deep Q Network

## Running
```bash
python main.py
```

## Benchmarks

lr = log10 of learning rate
e  = log10 of epsilon annealing steps
s  = sequence length per observation
Tq = Episodes between Q-syncronization
Tt = Steps between weight updates
Ts = Steps after which success is evident


 - If Success=FAIL, DQN variant has not been observed to succeed at the task
 - These are by no means optimal settings, just ones that are observed to work (i.e. it is very possible that the network could learn faster with different parameters.)

### Normal DQN

_All benchmarking in this section was done with a dense 3-layer neural network, with 400 x 400 x n_actions nodes._

| Environment    | Success  | lr  | e | s | Tq | Tt | Ts   |
| -------------- | -------- | --- | _ | _ | __ | __ | ____ |
| Debug-v0       | True     | -3  | 4 | 1 | 2  | 4  | 15 k |
| CartPole-v1    | True
