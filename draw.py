import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# attractor_interval = 6
# alpha = 0.7
#
#
# x = np.linspace(0, 40, 2000)
# attractor = (np.floor(x / attractor_interval) + 0.5) * attractor_interval
#
# target_lnr = attractor - alpha * (x - attractor)
# target_sin = x - alpha * attractor_interval * np.sin( ( 1 - 2 / attractor_interval * (x - attractor)) * np.pi )
#
#
# plt.plot(x, target_sin, color='blue')
# plt.title('output loss')
# plt.show()


logit1 = torch.tensor([10.0,9.5,9.8])

logit2 = torch.tensor([10.0,0.5,0.7])

logit3 = torch.tensor([10.0,1.0,1.0])

logit4 = torch.tensor([10.0,1.0,1.0])

logits1 = torch.stack( (logit1, logit2) ,0 )

logits2 = torch.stack( (logit3, logit4) ,0 )

print(logits1)
print(logits2)

for i in range(1):
    logits1 += logits2

logits1 /= 2

print(logits1)


