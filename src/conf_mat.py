from models import pred

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

def conf_mat(pred, true, model_folder, trial_ts):
    conf_arr = confusion_matrix(pred, true)
    print conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = max(sum(i, 0),1)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width, height = np.asarray(norm_conf).shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    #alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    plt.title(str(trial_ts)+'conf_matrix\naccuracy:'+str(np.sum(np.equal(pred, true))/float(len(pred))))
    plt.xticks(range(width))#, alphabet[:width])
    plt.yticks(range(height))#, alphabet[:height])
    plt.savefig(os.path.join(model_folder,'results','archive',trial_ts+'con_mat.png'), format='png')

if __name__=='__main__':
    model_folder = 'data'
    save=None
    trial_ts='20170506_005543'
    x_datapath='../data/X.pickle'
    y_datapath='../data/y.pickle'
    pred, true = pred(trial_ts=trial_ts, x_datapath=x_datapath, y_datapath=y_datapath, model_folder=model_folder, save=save)
    cm = conf_mat(true, pred, model_folder, trial_ts)
    print('accuracy:', np.sum(np.equal(pred, true))/float(len(pred)))
    print cm
