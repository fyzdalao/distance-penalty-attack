import victim
import utils




def load_model():
   return victim.Model()

def load_data(model, amount=2000, random_seed=0, need_right_prediction=False):
   #return utils.sample_imagenet(model, amount, random_seed, need_right_prediction)
   x_test, y_test = utils.sample_imagenet_every_class(model, random_seed, need_right_prediction=True)
   return x_test, y_test



if __name__ == '__main__':
   model = load_model()
   x_test, y_test = load_data(model=model)