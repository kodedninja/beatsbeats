# Audio Onset, Tempo and Beat Detection

For a course at uni.

## Usage

Training data for the onset detection (`.wav` and `.gt` files) goes to `data/train_onsets`, then 
the model can be trained by running the `train.ipynb` notebook from top to bottom.

For getting the submission file, `onsets.py`, `tempo.py` and `beats.py` can be run respectively.

## Description

As our strategy for solving the three challenges, we followed the approach of using a neural-
network architecture for the onset prediction, and then using the onset envelope we get from this
model as the input for the tempo and beats estimation, where we rely on classical, algorithmic
methods. Most of our efforts therefore were spent on pre-processing the data and trying out
different training solutions and network architectures for the onset detection challenge, while for
the latter two, we implemented solutions described on the slides, which didn’t require too much
experimentation.

#### Onset Detection

Our approach for training a neural-network for onset detection was based on [1], whose prominent
performance motivated us to use a ”deep” method for this challenge. We applied the same data
pre-processing as they did, namely transforming the audio signals into three logarithmically scaled
Mel-scaled spectograms with window sizes of 23ms, 46ms and 93ms and a hop size of 10ms. Further,
we normalized each frequency band to zero mean and unit variance. The spectograms were then
chopped into pieces of 15 frames (± 70ms) and fed into the neural network.
The most prominent difference between our model and the model used in [^1] was that while
their model classified only the frame at the center of the 15 frame piece as an onset or not, seeing
it as a binary classification problem, our model doesn’t reduce the input in length and classifies all
the 15 frames at the same time. Therefore the model directly learns to output an onset envelope
over time. To achieve this, we took the CNN architecture from [^1] and replaced the final fully-
connected layers with common solutions from modern vision models: global average / max pooling
and 1x1 convolutions. We also adapted the convolutional layers with paddings so that the input
doesn’t reduce in length. This model already learned to identify some onsets, but as most of
the parameters in the model from the paper were in the final two layers, our model needed more
complexity. Therefore, we scaled up and introduced residual blocks to achieve the final architecture
described below.

As targets, we used binary vectors of length 15, where most commonly a single element rep-
resented an onset and the others were negative examples. Because of this heavy imbalance, we
weighted positive frames 14x more than negative frames, which seemed to be essential not only to
speed up learning, but also to stop the model from classifying everything as negative. Additionally,
to counter the issue described in [^1] (”Some onsets have a soft attack, though, or are not annotated
with 10 ms precision, resulting in actual onsets being presented to the network as negative training
examples. To counter this, we would like to train on less sharply defined ground truth.”) we applied
the same solution, namely using ±1 frame around a positive target also as positive, but with a total
weight of 0.5.

Our final architecture looks the following: a 7x3 convolutional layer with 16 feature maps, a 3x3
residual block with 16 kernels, 3x1 max-pooling, a 3x3 convolutional layer with 32 feature maps, a
3x3 residual block with 32 kernels, 3x1 max-pooling, a final 3x3 convolutional layer with 64 feature
maps, global-max-pooling reducing the remaining 6 bands to 64 features with a single pixel and
a final 1x1 convolution reducing the 64 feature maps to a single dimension. After each pooling
operation we apply a dropout with p = 0.4 and use spatial batch norm in the residual blocks. As
an activation we use ReLU, except on the final outputs, where we use the sigmoid.
For classifying onsets we smooth the output we get by passing the whole spectogram through
the network by a Hamming window of 5 and then choose local maxima over a certain threshold.
We believe the model is far from its potential, as we didn’t have enough compute to perform a
hyperparameter search (and it was out of the scope as well).

#### Tempo Estimation

For estimating the tempo we apply the autocorrelation method described in the slides. We auto-
correlate the smoothed output signal of the onset detector with τ between 60 and 200 BPM. We
take the two highest peaks of this autocorrelation signal and report them as the tempo.

#### Beat Tracking

For beat tracking we put the two solutions from above together. Using the smoothed output of the
onset detector we estimate the most likely tempo (taking only the highest peak in the autocorrelation
signal) and using these we run the dynamic programming algorithm from [^2]. First we convolve
the onset signal with the given period and then run the DP algorithm on this signal, getting a
cumulative score for being a beat over every timepoint and their respective backlinks to the last
most prominent beat from there. Finally, we take the highest point of the cumulative score signal
and backtrace from there to the first beat using the backlinks.

Windowed beat estimation didn’t improve our score, and interestingly running the beat tracking
DP algorithm on the onset envelope directly without convolving it with the given period didn’t
worsen the performance.

[^1]: Jan Schlüter and Sebastian Böck. Improved Musical Onset Detection with Convolutional Neural Networks.
In Proceedings of the IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 6979–6983, 2014.

[^2]: ] Daniel P.W. Ellis. Beat Tracking by Dynamic Programming. 2017
https://www.ee.columbia.edu/ dpwe/pubs/Ellis07-beattrack.pdf