import cPickle as pickle
import numpy as np

with open('results/deepyeti/network-snapshot-008400.pkl', 'rb') as f:
  g1, _, g2 = pickle.load(f)

for net, netname in zip([g1, g2], ['g1', 'g2']):
  net_vars = []
  for param in net.trainable_params():
    param_name = param.name
    param_gpuarray = param.eval()
    param_numpy = np.array(param.eval())
    net_vars.append((param_name, param_numpy))

  with open('results/deepyeti/gvars_{}.pkl'.format(netname), 'wb') as f:
    pickle.dump(net_vars, f)
