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



