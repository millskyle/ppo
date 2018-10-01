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

If Success=FAIL, DQN variant has not been observed to succeed at the task

### Normal DQN

| Environment    | Success  | lr  | e | s | Tq | Tt | Ts   |
| -------------- | -------- | --- | _ | _ | __ | __ | ____ |
| Debug-v0       | True     | -3  | 4 | 1 | 2  | 4  | 15 k | 
| CartPole-v1    | 





