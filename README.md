# F-LM

Language modeling. This codebase contains implementation of G-LSTM and
F-LSTM cells from [1]. It also might contain some ongoing experiments.

This code was forked from https://github.com/rafaljozefowicz/lm and contains "BIGLSTM" language model baseline from [2].

Current code runs on Tensorflow r1.0 and supports multi-GPU data parallelism using synchronized gradient updates.

# Best perplexity
On One Billion Words benchmark using 8 GPUs in one DGX-1, BIG G-LSTM G4 was able to achieve 24.29 after 2 weeks of training and 23.36 after 3 weeks.

# Performance
Not using XLA optimization for now. To be tested.
(In all experiments minibatch of 128 per GPU is used)

* SMALLLSTM model on 1xGP100 is getting about ~34K wps.
* SMALLLSTM model on 2xGP100 is getting about ~54.9K wps.
* BIGLSTM model on 1xGP100 is getting about ~4.8K wps
* BIGLSTM model on 2xGP100 is getting about ~8.5K wps
* BIG G-LSTM G4 model on 2xGP100 is getting about ~17.4K wps
* BIG F-LSTM F512 model on 2xGP100 is getting about ~18.5K wps


On DGX-1, from [1], after 1 week of training on DGX-1 using all 8 GPUs.
(newer code should be faster).

| Model           | Perplexity | Steps      | WPS         |
| --------------- | :--------: | :--------: | :---------: |
| BIGLSTM         | 31.001     |    584.6K  |  20.3K      |
| BIG F-LSTM F512 | 28.11      |    1.217M  |  42.9K      |
| BIG G-LSTM G4   | 28.17      |    1.128M  |  41.7K      |
| BIG G-LSTM G16  | 34.789     |    850.4K  |  41.1K      |

Exact commit used to produce results from [1]: d98fb110053c187354caf68ff56f5a8535926b5d (should work with TF r1.0)

## Dependencies
* TensorFlow r1.0
* Python 2.7 (should work with Python 3 too)
* 1B Word Benchmark Dataset (https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark to get data)

## To run
Assuming the data directory is in: `/raid/okuchaiev/Data/LM1B/1-billion-word-language-modeling-benchmark-r13output/`, execute:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

SECONDS=604800
LOGSUFFIX=BIGLSTM

#train
python /home/okuchaiev/repos/f-lm/single_lm_train.py --logdir=/raid/okuchaiev/Workspace/LM/FGLSTM/$LOGSUFFIX --num_gpus=8 --datadir=/raid/okuchaiev/Data/LM1B/1-billion-word-language-modeling-benchmark-r13output/ --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=128  > train_$LOGSUFFIX.log 2>&1

#eval
python /home/okuchaiev/repos/f-lm/single_lm_train.py --logdir=/raid/okuchaiev/Workspace/LM/FGLSTM/$LOGSUFFIX --num_gpus=8 --datadir=/raid/okuchaiev/Data/LM1B/1-billion-word-language-modeling-benchmark-r13output/ --mode=eval_full --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=16 > eval_full_$LOGSUFFIX.log 2>&1
```
* To use G-LSTM cell specify ```num_of_groups``` parameter.
* To use F-LSTM cell specify ```fact_size``` parameter.

## To change hyper-parameters

The command accepts and additional argument `--hpconfig` which allows to override various hyper-parameters, including:
* batch_size=128 - batch size *per GPU*. Global batch size = batch_size*num_gpus
* num_steps=20 - number of LSTM cell timesteps
* num_shards=8 - embedding and softmax matrices are split into this many shards
* num_layers=1 - numer of LSTM layers
* learning_rate=0.2 - learning rate for optimizer
* max_grad_norm=10.0 -  maximum acceptable gradient norm for LSTM layers
* keep_prob=0.9 - dropout keep probability
* optimizer=0 - which optimizer to use: Adagrad(0), Momentum(1), Adam(2), RMSProp(3), SGD(4)
* vocab_size=793470 - vocabluary size
* emb_size=512 - size of the embedding (should be same as projected_size)
* state_size=2048 - LSTM cell size
* projected_size=512 - LSTM projection size
* num_sampled=8192 - training uses sampled softmax, number of samples)
* do_summaries=False - generate weight and grad stats for Tensorboard
* max_time=180 - max time (in seconds) to run
* fact_size - to use F-LSTM cell, this should be set to factor size
* num_of_groups=0 - to use G-LSTM cell, this should be set to number of groups
* save_model_every_min=30 - how often to checkpoint
* save_summary_every_min=16 - how often to save summaries
* use_residual=False - whether to use LSTM residual connections

## Feedback
Forked code and GLSTM/FLSTM cells: okuchaiev@nvidia.com.

## References
* [1] [Factorization tricks for LSTM networks](https://openreview.net/forum?id=ByxWXyNFg&noteId=ByxWXyNFg), ICLR 2017 workshop.
* [2] [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
