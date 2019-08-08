# F-LM

Language modeling. This codebase contains implementation of G-LSTM and
F-LSTM cells from [1]. It also might contain some ongoing experiments.

This code was forked from https://github.com/rafaljozefowicz/lm and contains "BIGLSTM" language model baseline from [2].

Current code runs on Tensorflow r1.5 and supports multi-GPU data parallelism using synchronized gradient updates.

# Perplexity
~~On One Billion Words benchmark using 8 GPUs in one DGX-1, BIG G-LSTM G4 was able to achieve 24.29 after 2 weeks of training and 23.36 after 3 weeks.~~

__On 02/06/2018 We found an issue with our experimental setup which makes perplexity numbers listed in the paper invalid.__

__See current numbers in the table below.__

On DGX Station, after 1 week of training using all 4 GPUs (Tesla V100) and batch size of 256 per GPU:

| Model           | Perplexity | Steps      | WPS         |
| --------------- | :--------: | :--------: | :---------: |
| BIGLSTM         |  35.1      |    ~0.99M  |  ~33.8K     |
| BIG F-LSTM F512 |  36.3      |    ~1.67M  |  ~56.5K     |
| BIG G-LSTM G4   |  40.6      |    ~1.65M  |  ~56K       |
| BIG G-LSTM G2   |  36        |    ~1.37M  |  ~47.1K     |
| BIG G-LSTM G8   |  39.4      |    ~1.7M   |  ~58.5      |


## Dependencies
* TensorFlow r1.5
* Python 2.7 (should work with Python 3 too)
* 1B Word Benchmark Dataset (https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark to get data)

## To run
Assuming the data directory is in: `/raid/okuchaiev/Data/LM1B/1-billion-word-language-modeling-benchmark-r13output/`, execute:

```
export CUDA_VISIBLE_DEVICES=0,1,2,3

SECONDS=604800
LOGSUFFIX=FLSTM-F512-1week

python /home/okuchaiev/repos/f-lm/single_lm_train.py --logdir=/raid/okuchaiev/Workspace/LM/GLSTM-G4/$LOGSUFFIX --num_gpus=4 --datadir=/raid/okuchaiev/Data/LM/LM1B/1-billion-word-language-modeling-benchmark-r13output/ --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=256,fact_size=512  >> train_$LOGSUFFIX.log 2>&1

python /home/okuchaiev/repos/f-lm/single_lm_train.py --logdir=/raid/okuchaiev/Workspace/LM/GLSTM-G4/$LOGSUFFIX --num_gpus=1 --mode=eval_full --datadir=/raid/okuchaiev/Data/LM/LM1B/1-billion-word-language-modeling-benchmark-r13output/ --hpconfig run_profiler=False,float16_rnn=False,max_time=$SECONDS,num_steps=20,num_shards=8,num_layers=2,learning_rate=0.2,max_grad_norm=1,keep_prob=0.9,emb_size=1024,projected_size=1024,state_size=8192,num_sampled=8192,batch_size=1,fact_size=512

```

* To use G-LSTM cell specify ```num_of_groups``` parameter.
* To use F-LSTM cell specify ```fact_size``` parameter.

Note, that current data reader may miss some tokens when constructing mini-batches which can have a minor effect on final perplexity.

**For most accurate results**, use batch_size=1 and num_steps=1 in evaluation. Thanks to [Ciprian](https://github.com/ciprian-chelba) for noticing this.

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
Forked code and GLSTM/FLSTM cells: okuchaiev@nvidia.com

## References
* [1] [Factorization tricks for LSTM networks](https://openreview.net/forum?id=ByxWXyNFg&noteId=ByxWXyNFg), ICLR 2017 workshop.
* [2] [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410)
