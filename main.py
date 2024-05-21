import victim
import utils
from utils import margin_loss



def load_model():
   return victim.Model()


def load_data(model, amount=2000, random_seed=0, need_right_prediction=False):
   return utils.sample_imagenet(model, amount, random_seed, need_right_prediction)
   #x_test, y_test = utils.sample_imagenet_every_class(model, random_seed, need_right_prediction=True)
   return x_test, y_test


def try_the_model(model, x_test, y_test):
   logits = model(x_test)
   margin = margin_loss(y_test, logits)
   acc = (margin > 0).sum() / x_test.shape[0]
   print(acc)



if __name__ == '__main__':
   model = load_model()
   x_test, y_test = load_data(model=model)
   try_the_model(model, x_test, y_test)