from models import *

def main():
    commit_hash = save_code()
    embeddings_path = 'data/chord2vec_30hr.npz'
    x_datapath='data/X.pickle'
    y_datapath='data/y.pickle'
    load_embeddings(embeddings_path=embeddings_path)
    load_data(x_datapath=x_datapath, y_datapath=y_datapath)
    embeddings=False
    run_experiment(**locals())

main()

