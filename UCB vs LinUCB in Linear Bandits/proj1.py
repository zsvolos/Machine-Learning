"""
To run, try `python3 proj1.py` under the same directory.
By default, it will generate two files i.e., `data.out` and `figure.pdf`.
`data.out` stores the regrets and `figure.pdf` is the output figure.
"""
import os
from abc import ABC, abstractmethod
import random
import math

from absl import app
from absl import logging
from absl import flags
import numpy as np

from utils import EmArm, draw, write_to_file

FLAGS = flags.FLAGS


# flag definitions
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_string('fig', 'figurecombined.pdf', 'file for generated figure')
flags.DEFINE_boolean('novar', True, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_integer('T', 1000, 'time horizon')
flags.DEFINE_integer('trials', 100, 'total number of trials')
flags.DEFINE_integer('freq', 50, 'frenquency to report the intermediate regrets')
flags.DEFINE_boolean('minimax', True, 'compute minimax regret')


class Bandit():
  """Bandit Class
  There are only two arms indexed by 1 and 2
  """

  def __init__(self, mu_1, mu_2):
    self.__mu_1, self.__mu_2 = mu_1, mu_2

  def init(self):
    self.tot_samples = 0

  def pull_arm(self, index):
    if index not in [1, 2]:
      logging.fatal('Wrong Arm Index!')
    self.tot_samples += 1
    if index == 1:
      return np.random.binomial(1, self.__mu_1, 1)[0]
    return np.random.binomial(1, self.__mu_2, 1)[0]

  def regret(self, rewards):
    return self.tot_samples * max(self.__mu_1, self.__mu_2) - rewards


class Learner(ABC):
  """Base Learner Class
  """

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def init(self):
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

  def init(self):
    pass

  def choice(self, time):
    return time % 2 + 1

  def update(self, reward, action):
    pass

class Greedy(Learner):

  @property
  def name(self):
    return 'Greedy'

  def init(self):
    self.arm1 = EmArm()
    self.arm2 = EmArm()
    
    pass
  def choice(self, time):

    if time == 1:
      
      return 1 
    elif time == 2:
      return 2 
    else:
      if self.arm1.rewards > self.arm2.rewards:
        return 1
      else: 
        return 2
    

  def update(self, reward, action):
    if action == 1: 
      self.arm1.update(reward)
    else:
      self.arm2.update(reward)
    
class EpsGreedy(Learner):

  def __init__(self, eps):
    self.__eps = eps

  @property
  def name(self):
    return 'Epsilon-Greedy'

  def init(self):
    
    self.arm1 = EmArm()
    self.arm2 = EmArm()

    

  def choice(self, time):
    if time == 1:
      return 1
    elif time == 2:
      return 2
    else:
      a = random.uniform(0,1)
      if a <= (1-self.__eps/time):
        if self.arm1.em_mean > self.arm2.em_mean:
          return 1 
        else:
          return 2
      else:
        b = random.uniform(0,1)
        if b <= 0.5:
          return 1 
        else:
          return 2
      

  def update(self, reward, action):
    if action == 1: 
          self.arm1.update(reward)
    else:
          self.arm2.update(reward)    



class ExploreThenCommit(Learner):

  def __init__(self, C):
    self.__C = C

  @property
  def name(self):
    return 'ExploreThenCommit'

  def init(self, t):
    ###############################################
    self.t = t
    self.tau = self.__C * (self.t**(2/3))
    self.arm1 = EmArm()
    self.arm2 = EmArm()
    ###############################################

  def choice(self, time):
    ###############################################
    if time <= self.tau / 2:
      return 1 
    elif time <= self.tau and time > self.tau/2 :
      return 2 
    else: 
      if self.arm1.em_mean >= self.arm2.em_mean:
        return 1
      else:
        return 2

    ###############################################

  def update(self, reward, action):
    if action == 1: 
      self.arm1.update(reward)
    else:
      self.arm2.update(reward)               


class UCB(Learner):

  def __init__(self, alpha):
    self.__alpha = alpha

  @property
  def name(self):
    return 'UCB'

  def init(self):
    ###############################################
    self.arm1 = EmArm()
    self.arm2 = EmArm()

    ###############################################

  def choice(self, time):
    ###############################################
    if time == 1:
      return 1
    elif time == 2:
      return 2 
    else:
      sqr1 = self.__alpha * ( ( 2* np.log(time-1) ) /self.arm1.pulls)**(1/2)
      sqr2 = self.__alpha * ( ( 2* np.log(time-1) ) /self.arm2.pulls)**(1/2)
      if self.arm1.em_mean + sqr1 >= self.arm2.em_mean+sqr2:
        return 1 
      else:
        return 2
    
    
    ###############################################

  def update(self, reward, action):
    if action == 1 : 
      self.arm1.update(reward)
    else:
      self.arm2.update(reward)



class TS(Learner):

  @property
  def name(self):
    return 'TS'

  def init(self):
    self.arm1 = EmArm()
    self.arm2 = EmArm()
    
  def choice(self, time):
    ###############################################
    xt = np.random.beta(1 + self.arm1.rewards , 1 + (self.arm1.pulls-self.arm1.rewards))
    yt = np.random.beta(1 + self.arm2.rewards , 1 + (self.arm2.pulls-self.arm2.rewards)) 
    if xt > yt:
      return 1
    else: 
      return 2
    ###############################################

  def update(self, reward, action):
    ###############################################
    if action == 1 :
      self.arm1.update(reward)
    else:
      self.arm2.update(reward)
    ###############################################


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
  np.random.seed(200)

  trials = FLAGS.trials
  freq = FLAGS.freq
  T = FLAGS.T

  # policies to be compared
  # add your methods here
  policies = [Uniform_Sampling(), Greedy(), EpsGreedy(1), ExploreThenCommit(1), UCB(.5), TS()]

  mus = [(0.4, 0.6)]
  if FLAGS.minimax:
    mus = [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.25, 0.75), (0.35, 0.65)]

  for policy in policies:
    logging.info('run policy %s' % policy.name)
    for trial in range(trials):
      if trial % 50 == 0:
        logging.info('trial: %d' % trial)

      minimax_regret = dict()
      for (mu_1, mu_2) in mus:
        bandit = Bandit(mu_1, mu_2)
        agg_regret = dict()
        if policy.name == 'ExploreThenCommit':
          for comp_t in range(0, T+1):
            if comp_t % freq == 0:
              if comp_t == 0:
                agg_regret[comp_t] = 0
              else:
                # initialization
                bandit.init(); policy.init(comp_t); rewards = 0
                for t in range(1, comp_t+1):
                  action = policy.choice(t)
                  reward = bandit.pull_arm(action)
                  policy.update(reward, action)
                  rewards += reward
                agg_regret[comp_t] = bandit.regret(rewards)
        else:
          # initialization
          bandit.init(); policy.init(); rewards = 0
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
