import logging
logging.basicConfig(level = logging.INFO , format=
        '%(asctime)s:%(levelname)s:%(name)s:%(threadName)s:line %(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

def save_conf_mat(pred, true, dirpath):
    conf_arr = confusion_matrix(pred, true)
    logger.info('generated confusion matrix: '+str(conf_arr))
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
    accuracy = np.sum(np.equal(pred, true))/float(len(pred))
    plt.title('confusion matrix\naccuracy:'+str(accuracy))
    plt.xticks(range(width))#, alphabet[:width])
    plt.yticks(range(height))#, alphabet[:height])
    plt.savefig(os.path.join(dirpath, 'confmat.png'), format='png')
    try:
        np.savez(os.path.join(dirpath, 'confmat.npz'), np.array(conf_arr))
    except Exception as e:
        logger.exception(e)
    return conf_arr


