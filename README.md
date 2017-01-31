# F-LM

This code was forked from https://github.com/rafaljozefowicz/lm 
Which implemented LSTM language model baseline from https://arxiv.org/abs/1602.02410
The code supports running on the machine with multiple GPUs using synchronized gradient updates (which is the main difference with the paper).

# Main modification from the forked code

* Research experimental GLSTM and DLSTM cells
* Refactor code to make fuller use of TF RNN APIs
* More fine grained summary tracking 
* full eval mode
* Updated to work with Tensorflow master branch (as of Jan 30, 2016)


# Performance
## Current Code:
(In all experiments minibatch of 128 per GPU is used)
SMALLLSTM model on 1xP100 is getting about ~33K wps and achieves full eval set perplexity of 41.413 after 24hours
BIGLSTM model on 1xP100 is getting about ~4.5K wps

BIGLSTM model 1 DGX-1, all 8 GPUs is getting ~20.3K wps (current code is likely to do a little better)


## Original code:
The code was tested on a box with 8 Geforce Titan X and LSTM-2048-512 (default configuration) can process up to 100k words per second.
The perplexity on the holdout set after 5 epochs is about 48.7 (vs 47.5 in the paper), which can be due to slightly different hyper-parameters.
It takes about 16 hours to reach these results on 8 Titan Xs. DGX-1 is about 30% faster on the baseline model.


## Dependencies
* TensorFlow master as of Jan 30, 2016
* Python 2.7
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

## To change hyper-parameters

The command accepts and additional argument `--hpconfig` which allows to override various hyper-parameters, including:
* batch_size=128 - batch size
* num_steps=20 - number of unrolled LSTM steps
* num_shards=8 -  embedding and softmax matrices are split into this many shards
* num_layers=1 - number of LSTM layers
* learning_rate=0.2 - learning rate for adagrad
* max_grad_norm=10.0 - maximum acceptable gradient norm 
* keep_prob=0.9 - for dropout between layers (here: 10% dropout before and after each LSTM layer)
* emb_size=512 - size of the embedding
* state_size=2048 - LSTM state size
* projected_size=512 - LSTM projection size 
* num_sampled=8192 - number of word target samples for IS objective during training
* dlayers=None - if this is not None, DLSTM cell will be used. Example: dlayers=512_512.
* do_layer_norm - dlayers must be not None. If True, will use layer normalization inside DNN inside DLSTM cell
* num_of_groups - if not 0, will use GLSTM cell with num_of_groups groups. dlayers must be None then.


## Feedback
Forked code and GLSTM/DLSTM cells: okuchaiev@nvidia.com
Original code/pape: rafjoz@gmail.com
