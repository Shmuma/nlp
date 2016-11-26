RNN Language model on PTB dataset
=================================

The story
---------
During taking excellent Stanford course CS224D taught by Richard Socher http://cs224d.stanford.edu/, I've
faced problem 2 from lab 2 (L2P2), which is about training simple RNN predicting next word in a sentence.

Solution of this problem need to be implemented using tensorflow, but usage of high-level RNN classes
is explicitly prohibited. It's good to get better understanding what's going on, but once you've got
understanding of internal machinery, it's not very useful for large, complicated models, like deep
bidirectional RNN or stacked LSTMs. After having working low-level solution,
I've realized that  there are lot to investigate there, for example:

1. Pre-training word embeddings (original version trains embeddings on the fly),
2. Use larger and deeper model,
3. Use different activation functions (in L2P2 sigmoid was used),
4. Play with LSTM/GRU,
5. Use more unrolling of RNN,
6. etc.

After playing with my L2P2 solution and wasting lots of time finding stupid bugs, I've decided to reimplement
the whole problem from scratch using latest functionality of TensorFlow.
