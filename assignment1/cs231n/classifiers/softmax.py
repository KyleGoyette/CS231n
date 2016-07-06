import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 # N = X.shape[0]
 # C=W.shape[1]
 # score = np.zeros((N,C))
 # probs=np.zeros_like(score)
#  for i in range(N):
#      for j in range(C):
#          #calc the scores (unnormalized log probabilities)
#          score[i,j] = X[i,j]*W[i,j]
#          #probs
#
  #score-=np.max(score,axis=0)
 # for i in range(N):
#      for j in range(C):
#          probs[i,j] = np.exp(score[i,j])
#
     # #calculate loss using softmax formula (numerator is correct class, denom is sum of all classes)
    #  loss+=-np.log(np.sum(probs[i,y[i]]/np.sum(probs[i,:])))
  #loss/=N
 # loss+=0.5*reg*np.sum(np.square(W))
#
 # for i in range(N):
#      for j in range(C):
#          #if correct class deriv is -X(1-normprob correct class)
#          if y[i]==j:
#              dW[i,j]=-X[i,j]*(1-(probs[i,j]/np.sum(probs[i,:])))
#          #if incorrect class gradient is -X*(-normprob of class)
#          else:
#              dW[i,j]=-X[i,j]*(-probs[i,j]/np.sum(probs[i,:]))

  #dW/=N
  #dW+=reg*W
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    maxscore = np.max(scores)
    shifted_scores = scores - maxscore
    probs = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores))
    loss += -np.log( probs[y[i]] )
    for j in xrange(num_classes):
        if y[i] == j:
            dW[:,j] += -X[i] * (1-probs[j])
        else:
             dW[:,j] += -X[i] * (-probs[j])
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N=X.shape[0]

  score=X.dot(W)
  maxscore=np.max(score,axis=1)
  score-= maxscore[:,np.newaxis]
  probs = np.exp(score)
  norm_probs = np.exp(score)/np.sum(np.exp(score),axis=1,keepdims=True)
  loss = -np.log(norm_probs[range(N),y])
  loss=loss.sum()
  loss/=N
  loss+=0.5*reg*(np.sum(W*W))


  dscore = norm_probs
  dscore[range(N),y]-=1
  dW= X.T.dot(dscore)

  dW/=N
  dW+=reg*W




  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
