from models import *


load_embeddings()

load_data()


model, params = get_model()


model.load_weights('data/models/current_20170425_210941/weights.h5')

transforms = [lambda x:sequence.pad_sequences(x, MAX_CHORDS), lambda y:y]

dm_valid = DataManager(valid, y_valid, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)
dm_test = DataManager(test, y_test, batch_size=c.batch_size, maxepochs=100*c.epochs+1, transforms=transforms)


y_valid_pred = model.predict_generator(dm_valid.batch_generator(), dm_valid.num_batches)
y_test_pred = model.predict_generator(dm_test.batch_generator(), dm_test.num_batches)


def pred(trial_ts='20170506_005543', x_datapath='../data/X.pickle', y_datapath='../data/y.pickle', model_folder='data', save=None, max_chords=5112):
    if not os.path.exists(x_datapath) or not os.path.exists(y_datapath):
        logger.info("data file doesn't exist: "+str(x_datapath)+' or '+str(y_datapath))
        return
    model_json_file = os.path.join(model_folder,'results/archive/', trial_ts + '_model.json')
    model_weights = os.path.join(model_folder, 'models/archive/', trial_ts + '_weights.h5')
    if os.path.exists(model_json_file) and os.path.exists(model_weights):
        model = model_from_json(open(model_json_file, 'r').read())
        model.load_weights(model_weights, by_name=False)
    else:
        logger.info("model json or weights do not exist: "+str(model_json_file)+' or: '+str(model_weights))
        return

    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    MAX_CHORDS=max_chords
    transforms = [lambda x:norm_pad(x, MAX_CHORDS), lambda y:y]
    dm_pred = DataManager(test, y_test, batch_size=c.batch_size, transforms=transforms)
    soft = model.predict_generator(generator=dm_pred.batch_generator(), steps=dm_pred.num_batches, verbose=1)
    pred = np.argmax(soft, axis=1)
    true = np.argmax(y_test, axis=1)
    if save is not None:
        with open(save, "w") as f:
            f.write("pred,true\n")
            for (p,t)in zip(pred,true):
                f.write(str(p)+","+str(t)+"\n")

    return pred, true

def conf_mat_main():
    model_folder = 'data'
    save=None
    trial_ts='20170506_005543'
    x_datapath='../data/X.pickle'
    y_datapath='../data/y.pickle'
    pred, true = pred(trial_ts=trial_ts, x_datapath=x_datapath, y_datapath=y_datapath, model_folder=model_folder, save=save)
    cm = conf_mat(true, pred, model_folder, trial_ts)
    print('accuracy:', np.sum(np.equal(pred, true))/float(len(pred)))
    print cm

def main()
    return

if __name__=='__main__':
    main()


