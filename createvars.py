import pickle
import imageio
import numpy as np

### Create Dictionaries to reduce output space dimension
sym2intf = open('symbols.csv')
sym2intf.readline()
sym2int = {}
i=0
for line in sym2intf:
    V = line.split(',')
    sym2int[int(V[0])] = i
    i = i+1

num_labels = i
int2sym = dict(map(reversed, sym2int.items()))

### Function to read HasYv2 Data into numpy arrays
def loadfile2data(path):
    file = open(path)
    file.readline()

    x = []
    y = []
    samples=0
    for line in file:
        V = line.split(',')
        W = imageio.imread('HASYv2/' + V[0])
        x.append(np.asarray(W[:,:,0]/255).reshape(32,32))
        y.append(np.eye(1,num_labels,sym2int[int(V[1])]).flatten())
        samples = samples + 1

    file.close()
    return x, y, samples

print('Loading data')
xtraina, ytraina, trainsamplesa = loadfile2data('./HASYv2/hasy-data-labels.csv')
print('Data loaded')


xtraina = np.asarray(xtraina)
ytraina = np.asarray(ytraina)
p = np.random.permutation(trainsamplesa)

xtraina = xtraina[p,:,:]
ytraina = ytraina[p,:]

file = open('traina.pickle', 'wb')
pickle.dump([xtraina, ytraina, trainsamplesa], file)