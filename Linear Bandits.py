
import os
import math
from abc import ABC, abstractmethod

from absl import app
from absl import logging
from absl import flags
import numpy as np

from utils import draw, write_to_file, sphere_sampling, EmArm

FLAGS = flags.FLAGS

# flag definitions
flags.DEFINE_string('out', 'dataretry.out', 'file for generated data')
flags.DEFINE_string('fig', 'figureretry.pdf', 'file for generated figure')
flags.DEFINE_boolean('novar', True, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_integer('T', 1000, 'time horizon')
flags.DEFINE_integer('trials', 50, 'total number of trials')
flags.DEFINE_integer('freq', 50,'frenquency to report the intermediate regrets')
flags.DEFINE_integer('armnum', 30, 'number of arms for for input')
flags.DEFINE_boolean('minimax', False, 'compute minimax regret')
flags.DEFINE_integer('inputnum',10, 'number of inputs used in computing minimax regret')


class GaussianArm():
  """gaussian arm"""

  def __init__(self, mu, sigma=1):
    self.__mu = mu
    self.__sigma = sigma

  @property
  def mean(self):
    return self.__mu

  @property
  def std(self):
    return self.__sigma

  def pull(self):
    """return a numpy array of stochastic rewards"""
    return np.random.normal(self.__mu, self.__sigma, 1)[0]


class LinearBandit():
  """Linear Bandit Class
  Arms are numbered by 0 to len(contexts)-1 by default.
  """

  def __init__(self, contexts, theta):
    if len(contexts) < 2:
      logging.fatal('The number of arms should be at least two!')
    if not isinstance(contexts, list):
      logging.fatal('Features should be given in a list!')
    self.__contexts = [np.array(context) for context in contexts]
    self.__theta = np.array(theta)
    for _, context in enumerate(self.__contexts):
      if context.shape != self.__theta.shape:
        logging.fatal('The context and theta dimensions are unequal!')
    arms = [GaussianArm(np.dot(context, theta)) for context in self.__contexts]
    self.__arms = arms
    self.__arm_num = len(arms)
    self.__best_arm_ind = max(
        [(tup[0], tup[1].mean) for tup in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_ind]

  def init(self):
    self.tot_samples = 0
    self.__max_rewards = 0

  def pull_arm(self, index):
    if index not in range(self.__arm_num):
      logging.fatal('Wrong arm index!')
    self.tot_samples += 1
    return self.__arms[index].pull()

  @property
  def contexts(self):
    return self.__contexts

  def regret(self, rewards):
    return self.tot_samples * self.__best_arm.mean - rewards


class Learner(ABC):
  """Base Learner Class
  """

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def init(self, contexts):
    pass

  @abstractmethod
  def choice(self, time):
    # return an arm to pull
    pass

  @abstractmethod
  def update(self, reward, action):
    pass


class Uniform_Sampling(Learner):
  """Sample each arm the same number of times
  """

  @property
  def name(self):
    return 'Uniform'

  def init(self, contexts):
    self.__contexts = contexts

  def choice(self, time):
    return time % len(self.__contexts)

  def update(self, reward, action):
    pass



class UCB(Learner):

  def __init__(self, alpha):
    self.__alpha = alpha

  @property
  def name(self):
    return 'UCB'

  def init(self, contexts):
    ###############################################
    self.__contexts = contexts
    self.__arms = []
    #self.__armPlayCount = []
    for i in range(0,len(contexts)):
      self.__arms.append(EmArm())
      #self.__armPlayCount(0)
    ###############################################

  def choice(self, time):
    ###############################################
    for i in range(len(self.__arms)):
      if self.__arms[i].pulls == 0:
        #print(str(i) + ' is pulled')
        #print( str(self.__arms[i-1].pulls)+' is the number of pulls of the previous')
        return i
    argmaxVal = 0.0
    argmaxInd = 0

    for i in range(len(self.__arms)):
      
      currMean = self.__arms[i].em_mean

      alphaSqr = self.__alpha * (2*np.log(time-1) / self.__arms[i].pulls) ** 0.5

      finalVal = currMean + alphaSqr

      if finalVal > argmaxVal:
        argmaxVal = finalVal
        argmaxInd = i

    return argmaxInd


  def update(self, reward, action):    
    self.__arms[action].update(reward)
    #print(self.__arms[action].pulls)
    #print(action)

#####################################################################################
class LinUCB(Learner): 

  def __init__(self, alpha, T):
    self.__alpha = alpha
    self.__T = T

  @property
  def name(self):
    return 'LinUCB'

  def init(self, contexts):
    self.__contexts = contexts
    self.__arms = []
    #self.__armPlayCount = []
    for i in range(0,len(contexts)):
      self.__arms.append(EmArm())
      #self.__armPlayCount(0)

    self.A = np.identity(3)
    self.b = np.zeros((3,1))
    self.total_plays = 0
    argmaxVal = 0.0
    argmaxInd = 0
    self.alpha = 0.5
  
    
    
  def choice(self, time):

    argmaxInd = 0 
    argmaxVal = 0
    for i in range(len(self.__arms)):
      theta = np.dot(np.transpose(self.A), self.b)
      x = self.__contexts[i]
      xt = np.transpose(x)
      Aminus = np.linalg.inv(self.A)
      sqr1 = self.alpha * ((np.dot(np.dot(xt,Aminus),x))**(1/2)) 
      sqr2 = ( np.log( 10 * FLAGS.T**2 ) )**(1/2)
      finalVal = np.dot(xt,theta) *sqr1 *sqr2
      
      if finalVal > argmaxVal:
        argmaxVal = finalVal
        argmaxInd = i

    return argmaxInd
    

  def update(self, reward, action):
    self.__arms[action].update(reward)
    self.A = self.A + np.dot(self.__contexts[action],np.transpose(self.__contexts[action]))
    self.b = self.b + np.dot(np.transpose(np.array([self.__contexts[action]])),reward)
    

def main(argv):
  del argv

  if FLAGS.rm:
    os.remove(FLAGS.out)
  else:
    if FLAGS.out in os.listdir('./'):
      logging.fatal(('%s is not empty. Make sure you have'
                     ' archived previously generated data. '
                     'Try --rm flag which will automatically'
                     ' delete previous data.') % FLAGS.out)

  # for reproducing purpose
  np.random.seed(100)

  trials = FLAGS.trials
  freq = FLAGS.freq
  T = FLAGS.T
  inputnum = FLAGS.inputnum if FLAGS.minimax else 1

  # policies to be compared
  # add your methods here
  policies = [Uniform_Sampling(), UCB(0.5), LinUCB(0.5, FLAGS.T)]

  for policy in policies:
    logging.info('run policy %s' % policy.name)
    for trial in range(trials):
      if trial % 50 == 0:
        logging.info('trial: %d' % trial)

      minimax_regret = dict()
      for _ in range(inputnum):
        contexts = list(sphere_sampling(3, FLAGS.armnum))
        theta = [1,0,0]
        bandit = LinearBandit(contexts, theta)
        agg_regret = dict()
        # initialization
        bandit.init(); policy.init(contexts); rewards = 0
        for t in range(0, T+1):
          if t > 0:
            action = policy.choice(t)
            reward = bandit.pull_arm(action)
            policy.update(reward, action)
            rewards += reward
          if t % freq == 0:
            agg_regret[t] = bandit.regret(rewards)
        for t in agg_regret:
          minimax_regret[t] = max(minimax_regret.get(t, 0), agg_regret[t])
      # output one trial result into the output file
      write_to_file(dict({policy.name: minimax_regret}))

  # generate the final figure
  draw()

if __name__ == '__main__':
  app.run(main)


  
