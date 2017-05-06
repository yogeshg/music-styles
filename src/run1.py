from models import *

def main():
    commit_hash = save_code()
    embeddings_path = 'data/chord2vec_30hr.npz'
    x_datapath='data/X.pickle'
    y_datapath='data/y.pickle'
    load_embeddings(embeddings_path=embeddings_path)
    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    use_embeddings = True
    use_embeddings=True
    dilated_convs=False
    pooling = 'max'
    filter_sizes=range(1,10)
    num_filters=50
    use_batch_normalization=False
    run_experiment(**locals())
    num_filters = 500
    filter_sizes=range(1,4)
    run_experiment(**locals())
    filter_sizes=range(1,10)
    run_experiment(**locals())

if __name__ == '__main__':
    main()
