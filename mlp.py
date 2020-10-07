import numpy as np

def random_normal_weight_init(input, output):
  return np.random.normal(0,1,(output,input))

def random_weight_init(input,output):
  b = np.sqrt(6)/np.sqrt(input+output)
  return np.random.uniform(-b,b,(output,input))

def zeros_bias_init(outd):
  return np.zeros((outd,1))

def labels2onehot(labels):
  return np.array([[i==lab for i in range(10)]for lab in labels])

class Transform:
  """
  This is the base class. It does not need to be filled out.
  A Transform class represents a (one input, one output) function done to some input x
    In symbolic terms,
    if f represents the transformation, out is the output, x is the input,
      out = f(x)
  The forward operation is called via the forward function
  The derivative of the operation is computed by calling the backward function
    In ML, we compute the loss as a function of the output, so
    there is some function q that takes in the output of this transformation
    and outputs the loss
      Loss = q(out) = q(f(x))
    Hence, by chain rule (using the notation discussed in recitation),
      grad_wrt_x = (dout_dx)^T @ grad_wrt_out (@ is matrix multiply)
    This is useful in backpropogation.
  """
  def __init__(self):
    # this function in a child class used for initializing any parameters
    pass

  def forward(self, x):
    # x should be passed as column vectors
    pass

  def backward(self,grad_wrt_out):
    # this function used to compute the gradient wrt the parameters
    # and also to return the grad_wrt_x
    #   which will be the grad_wrt_out for the Transform that provided the x
    pass

  def step(self):
    # this function used to update the parameters
    pass

  def zerograd(self):
    # after you updated you might want to zero the gradient
    pass

class Identity(Transform):
  """
  Identity Transform
  This exists to give you an idea for how to fill out the template
  """
  def __init__(self,dropout_chance=0):
    Transform.__init__(self)

  def forward(self, x, train=True):
    self.shape = x.shape
    return x

  def backward(self,grad_wrt_out):
    return np.ones(self.shape) * grad_wrt_out

class ReLU(Transform):
  """
  ReLU non-linearity
  IMPORTANT the Autograder assumes these function signatures
  """
  def __init__(self,dropout_chance=0):
    Transform.__init__(self)
    self.dropout_chance = dropout_chance

  def forward(self, x, train=True):  
    
    if train:
      mask = np.random.uniform(0,1,x.shape) > self.dropout_chance
      mask_out = mask * x
    else:
      mask_out = (1 - self.dropout_chance) * x
    
    relu_forward = np.maximum(0,mask_out)
    self.x = mask_out
    return relu_forward
    
    # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function

  def backward(self,grad_wrt_out):
    backward_out = self.x>0
    backward_out = (backward_out*grad_wrt_out)
    return backward_out

class LinearMap(Transform):
  # This represents the matrix multiplication step
  # IMPORTANT the Autograder assumes these function signatures
  def __init__(self,indim,outdim,alpha=0,lr=0.01):
    Transform.__init__(self)
    self.lr = lr
    self.outdim = outdim
    self.indim = indim
    self.g_m_w = np.zeros((self.outdim, self.indim))
    self.g_m_b = np.zeros((self.outdim, 1))
    self.alpha = alpha

  def forward(self,x):
    #assumes x is a batch of column vectors (ie the shape is (indim,batch_size))
    #return shape (outdim,batch)
   
    self.x = x
    out = np.dot(self.w,x) + self.b
    return out

  def zerograd(self):
    self.gradient_input = 0
    self.gradient_bias = 0
    self.gradient_weights = 0

  def backward(self,grad_wrt_out):
    #assumes grad_wrt_out is in shape (outdim,batch)
    #return shape indim,batch

    self.grad_input = np.dot(self.w.T, grad_wrt_out)
    self.grad_weights = np.dot(grad_wrt_out, self.x.T)
    self.grad_bias = np.sum(
    grad_wrt_out, axis=1).reshape((self.outdim, 1))
    return self.grad_input

  def step(self):
    self.g_m_w = self.alpha * self.g_m_w - self.lr * self.grad_weights
    self.g_m_b = self.alpha * self.g_m_b - self.lr * self.grad_bias
    
    self.w += self.g_m_w
    self.b += self.g_m_b
    

  def getW(self):
    return self.w
    #return the weights

  def getb(self):
    return self.b
    #return the bias

  def loadparams(self,w,b):
    # IMPORTANT this function is called by the autograder
    # to set the weights and bias of this LinearMap
    self.w = w
    self.b = b

class SoftmaxCrossEntropyLoss:
  """
  Softmax Cross Entropy loss
  IMPORTANT the Autograder assumes these function signatures
  """
  def forward(self, logits, labels):
    # assumes the labels are one-hot encoded
    # assumes both logits and labels have shape (num_classes,batch_size)
    # returns scalar
    self.labels = labels
    logits_max = np.max(logits, axis=0) *(-1)
    self.exps = np.exp(logits + logits_max)
    self.sum_of_exps = np.sum(self.exps, axis=0)
    self.softmax = self.exps/self.sum_of_exps
    self.softmax[self.softmax == 0] = 1e-16
    self.loss = -np.sum(np.multiply(labels,np.log(self.softmax)))/logits[0].shape[0]
    return self.loss

  def backward(self):
    #return shape (num_classes,batch_size)
    gradient = self.softmax-self.labels
    gradient = gradient/self.labels.shape[1]
    return gradient

class Pipe(Transform):
  # Represents a flat data pipe.
  # The pipe forwards the data through each of its transforms in a queue
  # You don't have to use this module, but it makes it easier to implement the final MLP class
  def __init__(self,transq):
    Transform.__init__(self)
    self.transq = transq

  def forward(self,x):
    for t in self.transq:
      x = t.forward(x)
    return x

  def backward(self,grad_wrt_out):
    pass

  def zerograd(self):
    pass

  def step(self):
    pass

class SingleLayerSingleTaskMLP:
  """
  This MLP has one hidden layer
  IMPORTANT the Autograder assumes these function signatures
  """
  def __init__(self,inp,outp,hiddenlayer=100,alpha=0.1,dropout_chance=0,lr=0.01):
    Transform.__init__(self)
    self.inp = inp
    self.outp = outp
    self.hiddenlayer = hiddenlayer
    self.alpha = alpha
    self.dropout_chance = dropout_chance
    self.lr = lr
    self.LinearMap1 = LinearMap(self.inp,self.hiddenlayer,self.alpha,self.lr)
    self.LinearMap2 = LinearMap(self.hiddenlayer,self.outp,self.alpha,self.lr)
    self.ReLU = ReLU(self.dropout_chance)

  def forward(self,x):
    a = self.LinearMap1.forward(x)
    b = self.ReLU.forward(a)
    c = self.LinearMap2.forward(b)
    return c

    # x has shape

  def backward(self,grad_wrt_out):
    a = self.LinearMap2.backward(grad_wrt_out)
    b = self.ReLU.backward(a)
    c = self.LinearMap1.backward(b)
    return c

  def zerograd(self):
    self.LinearMap1.zerograd()
    self.LinearMap2.zerograd()

  def step(self):
    self.LinearMap1.step()
    self.LinearMap2.step()

  def loadparams(self,Ws,bs):
    # Ws is a length two list, representing the weights for the first LinearMap and the second LinearMap
    # ie Ws == [LinearMap1.W, LinearMap2.W]
    # bs is the bias for the layers, in the same order

    self.LinearMap1_w = Ws[0]
    self.LinearMap2_w = Ws[1]
    self.LinearMap1_b = bs[0]
    self.LinearMap2_b = bs[1]

    self.LinearMap1.loadparams(self.LinearMap1_w,self.LinearMap1_b)
    self.LinearMap2.loadparams(self.LinearMap2_w,self.LinearMap2_b)


  def getWs(self):
    return [self.LinearMap1_w, self.LinearMap2_w]
    # return the weights in a list, ordered [LinearMap1.W, LinearMap2.W]
   

  def getbs(self):
    return [self.LinearMap1_b, self.LinearMap2_b]
    # return the bias in a list, ordered [LinearMap1.b, LinearMap2.b]

class TwoLayerSingleTaskMLP:
  def __init__(self,inp,outp,hiddenlayers=[100,100],alpha=0.1,dropout_chance=0,lr=0.01):
    Transform.__init__(self)
    self.inp = inp
    self.outp = outp
    self.hiddenlayers = hiddenlayers
    self.alpha = alpha
    self.dropout_chance = dropout_chance
    self.lr = lr

    self.LinearMap1 = LinearMap(self.inp,self.hiddenlayers[0],self.alpha,self.lr)
    self.LinearMap2 = LinearMap(self.hiddenlayers[0],self.hiddenlayers[1],self.alpha,self.lr)
    self.LinearMap3 = LinearMap(self.hiddenlayers[1],self.outp,self.alpha,self.lr)
    self.ReLU1 = ReLU(self.dropout_chance)
    self.ReLU2 = ReLU(self.dropout_chance)


  def forward(self,x):
    a = self.LinearMap1.forward(x)
    b = self.ReLU1.forward(a)
    c = self.LinearMap2.forward(b)
    d = self.ReLU2.forward(c)
    e = self.LinearMap3.forward(d)
    return e


  def backward(self,grad_wrt_out):
    a = self.LinearMap3.backward(grad_wrt_out)
    b = self.ReLU2.backward(a)
    c = self.LinearMap2.backward(b)
    d = self.ReLU1.backward(c)
    e = self.LinearMap1.backward(d)
    return e


  def zerograd(self):
    self.LinearMap1.zerograd()
    self.LinearMap2.zerograd()
    self.LinearMap3.zerograd()


  def step(self):
    self.LinearMap1.step()
    self.LinearMap2.step()
    self.LinearMap3.step()
    

  def loadparams(self,Ws,bs):

    self.LinearMap1_w = Ws[0]
    self.LinearMap2_w = Ws[1]
    self.LinearMap3_w = Ws[2]

    self.LinearMap1_b = bs[0]
    self.LinearMap2_b = bs[1]
    self.LinearMap3_b = bs[2]

    self.LinearMap1.loadparams(self.LinearMap1_w,self.LinearMap1_b)
    self.LinearMap2.loadparams(self.LinearMap2_w,self.LinearMap2_b)
    self.LinearMap3.loadparams(self.LinearMap3_w,self.LinearMap3_b)


    # Ws is a length three list, representing the weights for the first LinearMap, the second LinearMap, and the third
    # ie Ws == [LinearMap1.W, LinearMap2.W, LinearMap3.W]
    # bs is the bias for the layers, in the same order

  def getWs(self):
    return [self.LinearMap1_w, self.LinearMap2_w, self.LinearMap3_w]
    # return the weights in a list, ordered [LinearMap1.W, LinearMap2.W, LinearMap3.W]

  def getbs(self):
    return [self.LinearMap1_b, self.LinearMap2_b, self.LinearMap3_b]
    # return the weights in a list, ordered [LinearMap1.b, LinearMap2.b, LinearMap3.b]
 

class TwoLayerTwoTaskMLP:
  def __init__(self,inp,outp,hiddenlayers=[100,100],alpha=0.1,dropout_chance=0,lr=0.01):
    self.inp = inp
    self.outp = outp
    self.hiddenlayers = hiddenlayers
    self.alpha = alpha
    self.dropout_chance = dropout_chance
    self.lr = lr

    self.LinearMap1 = LinearMap(self.inp,self.hiddenlayers[0],self.alpha,self.lr)
    self.LinearMap2 = LinearMap(self.hiddenlayers[0],self.hiddenlayers[1],self.alpha,self.lr)
    self.LinearMap3 = LinearMap(self.hiddenlayers[1],self.outp,self.alpha,self.lr)
    self.LinearMap4 = LinearMap(self.hiddenlayers[1],self.outp,self.alpha,self.lr)

    self.ReLU1 = ReLU(self.dropout_chance)
    self.ReLU2 = ReLU(self.dropout_chance)

  def forward(self,x):
    a = self.LinearMap1.forward(x)
    b = self.ReLU1.forward(a)
    c = self.LinearMap2.forward(b)
    d = self.ReLU2.forward(c)
    e1 = self.LinearMap3.forward(d)
    e2 = self.LinearMap4.forward(d)
    return e1,e2
    # returns two numbers

  def backward(self,grad_wrt_out,loss_weights=[1,1]):
    a1 = self.LinearMap3.backward(grad_wrt_out[0])
    a2 = self.LinearMap4.backward(grad_wrt_out[1])
    b = self.ReLU2.backward(loss_weights[0]*a1 + loss_weights[1]*a2)
    c = self.LinearMap2.backward(b)
    d = self.ReLU1.backward(c)
    e = self.LinearMap1.backward(d)
    return e


  def zerograd(self):
    self.LinearMap1.zerograd()
    self.LinearMap2.zerograd()
    self.LinearMap3.zerograd()
    self.LinearMap4.zerograd()

  
  def step(self):
    self.LinearMap1.step()
    self.LinearMap2.step()
    self.LinearMap3.step()
    self.LinearMap4.step()


  def loadparams(self,Ws,bs):
    self.LinearMap1_w = Ws[0]
    self.LinearMap2_w = Ws[1]
    self.LinearMap3_w = Ws[2]
    self.LinearMap4_w = Ws[3]

    self.LinearMap1_b = bs[0]
    self.LinearMap2_b = bs[1]
    self.LinearMap3_b = bs[2]
    self.LinearMap4_b = bs[3]

    self.LinearMap1.loadparams(self.LinearMap1_w,self.LinearMap1_b)
    self.LinearMap2.loadparams(self.LinearMap2_w,self.LinearMap2_b)
    self.LinearMap3.loadparams(self.LinearMap3_w,self.LinearMap3_b)
    self.LinearMap4.loadparams(self.LinearMap4_w,self.LinearMap4_b)

    # Ws is a length three list, representing the weights
    # ie Ws == [LinearMap1.W, LinearMap2.W, LinearMap3.W, LinearMap4.W]
    # where LinearMap3 is directly responsible for outputting the first number from self.forward(x)
    # and LinearMap4 is directly responsible for the second
    # bs is the bias for the layers, in the same order


  def getWs(self):
    return [self.LinearMap1_w, self.LinearMap2_w, self.LinearMap3_w,self.LinearMap4_w]

    # return the weights in a list, ordered [LinearMap1.W, LinearMap2.W, LinearMap3.W,LinearMap4.W]
   
  def getbs(self):
    return [self.LinearMap1_b, self.LinearMap2_b, self.LinearMap3_b,self.LinearMap4_b]
    # return the weights in a list, ordered [LinearMap1.b, LinearMap2.b, LinearMap3.b,LinearMap4.b]
    