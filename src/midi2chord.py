import sys
import cPickle
import argparse
import logging
import os
import subprocess
import json
import urllib
import tarfile
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_csv(filename):
    import csv
    data=[]
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

def ignore_track(data):
    ignore=[]
    for i in range(len(data)):
        if data[i][2]=="Program_c" and data[i][4] <= 88:
            ignore.append(data[i][0])
    return ignore

def build_chords(data, ignore_tracks, noclip):
    sorted_data = sorted(data, key=lambda x: int(x[1]))
    if noclip:
        clip_min=-1
        clip_max=1000
    else:
        clip_min=9
        clip_max=96
    current_chord = []
    chords = []
    p_track = None
    p_tic = None
    p_type = None
    p_note = None
    p_vel = None

    for i in range(len(sorted_data)):
        prev_mes=sorted_data[i-1]
        mes=sorted_data[i]
        try:
            c_track = int(mes[0].strip())
            c_tic = int(mes[1].strip())
            c_type = mes[2].strip()
            c_chan = int(mes[3].strip())
            c_note = int(mes[4].strip())
            c_vel = int(mes[5].strip())
        except:
            continue
        if c_track not in ignore_tracks:
            if c_type=='Note_on_c' and int(c_vel)!=0 and c_note>=clip_min and c_note<=clip_max and c_chan!=9:
                if c_tic==p_tic:
                    current_chord.append(c_note-max(0,clip_min))
                else:
                    chords.append(current_chord[:])
                    current_chord.append(c_note)
            elif (c_type=='Note_on_c' and int(c_vel)==0) or c_type=='Note_off_c':
                if c_tic==p_tic:
                    try:
                        current_chord.remove(c_note)
                    except:
                        pass
                else:
                    chords.append(current_chord[:])
                    try:
                        current_chord.remove(c_note)
                    except:
                        pass
        p_track = c_track
        p_tic = c_tic
        p_type = c_type
        p_note = c_note
        p_vel = c_vel
    
    chords = filter(None, chords)

    return chords

def msd_id_to_dirs(l_root,msd_id, save_dir):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    p = os.path.join(l_root, msd_id[2], msd_id[3], msd_id[4], msd_id)
    if os.path.exists(p):
        for f in os.listdir(p):
            if f.endswith(".mid"):
                l_id, csv_file_path = midi2csv(p,f, save_dir)
                yield( (l_id, csv_file_path) )
                deleteCsv(csv_file_path)
    return

def allcsvs(l_root, msd_id, save_dir):
    for l_id, csv in msd_id_to_dirs(l_root, msd_id, save_dir):
        yield (l_id, csv)

def midi2csv(p,midifname, save_dir):
    path = os.path.join(p,midifname)
    csv_file_path = os.path.join(save_dir, '/tmp/test.csv')
    l_id = midifname
    bashCommand = "midicsv %s %s" % (path, csv_file_path)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return l_id, csv_file_path

def deleteCsv(csvfname):
    #deletes csvfname
    bashCommand = "rm %s" % csvfname
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return

def loadLabels(strat, genres):

    d = {}
    labels = []
    with open(strat) as f:
        for line in f:
            if line[0]!="%":
                (key, val) = line.split()
                d[key] = val
    with open(genres) as f:
        for line in f:
            if line[0]!="%":
                (m_id, genre) = line.split("\t")
                try:
                    labels.append((m_id, genre, d[m_id]))
                except:
                    pass
    return labels

def classPkl(data_root, save, noclip=True, valid_cut = 0.2, number_files=100):
    checkDataExist(data_root, save, classification=True)
    stratSplit=os.path.join(data_root,'msd-topMAGD-partition_percentageSplit_0.8-v1.0.cls')
    genres=os.path.join(data_root,'msd-topMAGD-genreAssignment.cls')
    l_root=os.path.join(data_root,'lmd_aligned')
    trainSongs=[]
    testSongs=[]
    trainLabels=[]
    testLabels=[]
    trainMeta=[]
    testMeta=[]
    i=1
    for (m_id, genre, isTest) in loadLabels(stratSplit, genres):
        for l_id, csvname in allcsvs(l_root, m_id, save):
            # at this point csvname will exist
            data = load_csv(csvname)
            ignore = ignore_track(data)
            chords = build_chords(data, ignore, noclip)
            if isTest == 'TRAIN':
                trainSongs.append(chords)
                trainLabels.append(genre)
                trainMeta.append([m_id, l_id, genre, isTest])
            else:
                testSongs.append(chords)
                testLabels.append(genre)
                testMeta.append([m_id, l_id, genre, isTest])
            if i % 50==1:
                print 'processing file', i
            i+=1
        if i>number_files:
            break
    
    l = len(trainSongs)
    l1 = int((1-valid_cut) * l)
    data = {'train': trainSongs[:l1], 'valid':trainSongs[l1:], 'test':testSongs}
    labels = {'train': trainLabels[:l1], 'valid': trainLabels[l1:], 'test':testLabels}
    meta = {'train': trainMeta[:l1], 'valid': trainMeta[:l1], 'test':testMeta}
        
    if save:
        print 'saving pkl and json files to', save
        with open(os.path.join(save,'X.pickle'),'wb') as f:
            cPickle.dump(data, f)
        with open(os.path.join(save,'y.pickle'),'wb') as f:
            cPickle.dump(labels, f)
        with open(os.path.join(save,'metadata.json'),'w') as f:
            json.dump(meta, f)

    return data, labels, meta

def checkDataExist(data_root, save, classification):
    if not os.path.exists(data_root):
        print "making data_root"
        os.makedirs(data_root)
    if not os.path.exists(save):
        print "making save"
        os.makedirs(save)
    if not os.path.exists(os.path.join(save,'tmp')):
        print "makeing tmp"
        os.makedirs(os.path.join(save,'tmp'))
    l_path = os.path.join(data_root, 'lmd_aligned')
    if not os.path.exists(l_path):
        lzip= os.path.join(data_root, 'lmd_aligned.tar.gz')
        if not os.path.exists(lzip):
            print 'lakh database zip doesnt exist in root data downloading it'
            urllib.urlretrieve('http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz', lzip)
        print 'please go to the directory %s and unzip lmd_aligned.tar.gz before running the program again' % data_root
        sys.exit()
    if classification:
        part_path = os.path.join(data_root,'msd-topMAGD-partition_percentageSplit_0.8-v1.0.cls')
        genre_path = os.path.join(data_root,'msd-topMAGD-genreAssignment.cls')
        if not os.path.exists(genre_path):
            print 'genre labels file doesnt exist downloading it:'
            urllib.urlretrieve('http://ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-genreAssignment.cls', genre_path)
        if not os.path.exists(part_path):
            print 'stratified split file doesnt exist downloading it:'
            urllib.urlretrieve('http://ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-partition_percentageSplit_0.8-v1.0.cls', part_path)
    print 'done ensuring data set up properly'



def embeddingPkl(data_root, save=None, noclip=True, train_cut=0.8, valid_cut=0.2, number_files=100):
    logger.debug(str(locals()))
    checkDataExist(data_root, save, classification=False)
    l_root=os.path.join(data_root,'lmd_aligned')
    songs=[]
    metaData=[]
    i=1
    for m_id in loadChords(stratSplit, genres):
        for l_id, csvname in allcsvs(l_root, m_id, save):
            # at this point csvname will exist
            data = load_csv(csvname)
            ignore = ignore_track(data)
            chords = build_chords(data, ignore, noclip)
            songs.append(chords)
            metaData.append([m_id, l_id])
            if i%50==1:
                print 'processing file', i
            i+=1
        if i>number_files:
            break

    l = len(songs)
    l1 = int(train_cut * l * (1-valid_cut))
    l2 = int(train_cut * l)
    data = {'train': songs[:l1], 'valid':songs[l1:l2], 'test':songs[l2:]}
    meta = {'train': metaData[:l1], 'valid':metaData[l1:l2], 'test':metaData[l2:]}
        
    if save:
        print 'saving pkl and json files to', save
        with open(os.path.join(save,'X.pickle'),'wb') as f:
            cPickle.dump(data, f)
        with open(os.path.join(save,'metadata.json'),'w') as f:
            json.dump(meta, f)

    return data, meta

def main(data_root, save=None, noclip=True, train_cut=0.6, valid_cut = 0.2, number_files=100, embedding=False):
    if embedding:
        sys.exit()
        embeddingPkl(data_root, save, noclip, train_cut, valid_cut, number_files)
    else:
        classPkl(data_root, save, noclip, valid_cut, number_files)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, type=str, help='specify the directory containing the lakh dataset, the labels and the stratified split')
    parser.add_argument('--save', required=True, type=str, help='specify what directory to save the outputs to')
    parser.add_argument('--noclip', default=False, action='store_true', help='use this flag to not clip the range of notes in chords to 88 range of chord2vec')
    parser.add_argument('--train_cut', type=float, default=0.6, help='fraction of data to be used for training')
    parser.add_argument('--valid_cut', type=float, default=0.2, help='fraction of training data to be used for validation')
    parser.add_argument('--number_files', type=int, default=100, help='the approximate number of data examples to create')
    parser.add_argument('--embedding', default=False, action='store_true', help='use this flag to save in the format for embedding training')
    args = parser.parse_args(sys.argv[1:])
    
    main(**args.__dict__)
