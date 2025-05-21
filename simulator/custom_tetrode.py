import yaml, os
import MEAutility as mu
import matplotlib.pyplot as plt

user_info = {'dim': 2,
             'electrode_name': 'tetrode_custom',
             'description': "customized tetrode",
             'pitch': 13, # 13 um
             'shape': 'circle',
             'size': 6.5,
             'sortlist': None,
             'stagger': 0,
             'type': 'mea'}

with open('tetrode_custom.yaml', 'w') as f:
    yaml.dump(user_info, f)

yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]

mu.return_mea()

mu.add_mea('tetrode_custom.yaml')

tetrode = mu.return_mea('tetrode_custom')
# plt.plot(tetrode.positions[:,1], tetrode.positions[:,0], 'o')
# _ = plt.axis('equal')
# mu.plot_probe(tetrode)
# plt.savefig('tetrode_custom.png')