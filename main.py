import victim
import utils
from utils import *
from square import square_attack_l2
from square import square_attack_linf
import numpy as np

def load_undefended_model():
   return victim.Model(defense='None')

def load_model(defense_type='None'):
   return victim.Model(defense=defense_type)


def load_data(model, amount=2000, random_seed=0, need_right_prediction=False):
   #return utils.sample_imagenet(model, amount, random_seed, need_right_prediction)
   return utils.sample_imagenet_every_class(model, random_seed, need_right_prediction=True)


def try_the_model(model, x_test, y_test):
   logits = model(x_test)
   margin = margin_loss(y_test, logits)
   acc = (margin > 0).sum() / x_test.shape[0]
   print(acc)

   # margin = margin_loss_their(y_test, logits)
   # margin = margin.min(1)
   # acc = (margin > 0).sum() / x_test.shape[0]
   # print(acc)


if __name__ == '__main__':
   undefended_model = load_undefended_model()
   x_test, y_test = load_data(model=undefended_model)

   np.random.seed(19260817)

   model = load_model(defense_type='inRND')

   #try_the_model(model, x_test, y_test)

   logits_clean = model(x_test)
   margin = margin_loss_their(y_test, logits_clean)
   correct_idx = margin > 0
   correct_idx = correct_idx.reshape((-1,))

   square_attack_linf(model=model, x=x_test, y=y_test, correct=correct_idx, n_iters=2500, eps=0.05, p_init=0.05, attack_tactic='average')

