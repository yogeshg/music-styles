import sys
import cPickle
import argparse
import logging
import os
import subprocess
import json
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

def msd_id_to_dirs(l_root,msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    p = os.path.join(l_root, msd_id[2], msd_id[3], msd_id[4], msd_id)
    if os.path.exists(p):
        for f in os.listdir(p):
            if f.endswith(".mid"):
                print 'BEFORE'
                l_id, csv_file_path = midi2csv(p,f)
                yield( (l_id, csv_file_path) )
                print 'AFTER'
                print "should print delets file"
                deleteCsv(csv_file_path)
    return

def allcsvs(l_root, msd_id):
    for l_id, csv in msd_id_to_dirs(l_root, msd_id):
        yield (l_id, csv)

def midi2csv(p,midifname):
    path = os.path.join(p,midifname)
    csv_file_path = '../data/tmp/test.csv'
    l_id = midifname
    bashCommand = "./midicsv %s ../data/tmp/test.csv" % path
    print bashCommand
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print 'creates file', midifname
    return l_id, csv_file_path

def deleteCsv(csvfname):
    #deletes csvfname
    bashCommand = "rm %s" % csvfname
    print bashCommand
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print 'deletes file'
    return

def loadLabels(stratSplit, genres):

    d = {}
    labels = []
    with open(stratSplit) as f:
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

def classPkl(stratSplit='../data/msd-topMAGD-partition_percentageSplit_0.8-v1.0.cls', genres='../data/msd-topMAGD-genreAssignment.cls', l_root='../data/lmd_aligned', noclip=True, train_cut=0.8, save='../data/tmp'):
    trainSongs=[]
    testSongs=[]
    trainLabels=[]
    testLabels=[]
    trainMeta=[]
    testMeta=[]
    i=1
    for (m_id, genre, isTest) in loadLabels(stratSplit, genres):
        for l_id, csvname in allcsvs(l_root, m_id):
            # at this point csvname will exist
            print csvname
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
            i+=1
    l = len(trainSongs)
    l1 = int(train_cut * l)
    data = {'train': trainSongs[:l1], 'valid':trainSongs[l1:], 'test':testSongs}
    labels = {'train': trainLabels[:l1], 'valid': trainLabels[l1:], 'test':testLabels}
    meta = {'train': trainMeta[:l1], 'valid': trainMeta[:l1], 'test':testMeta}
        
    if save:
        with open(os.path.join(save,'X.pickle'),'wb') as f:
            cPickle.dump(data, f)
        with open(os.path.join(save,'y.pickle'),'wb') as f:
            cPickle.dump(labels, f)
        with open(os.path.join(save,'metadata.json'),'w') as f:
            json.dump(meta, f)

    return data, labels, meta

        


def all_csv(directory):
    csvs=[]
    for root, dir_list, files in os.walk(directory):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))
    return csvs


def main(directory, save=None, noclip=True, train_cut=0.6, valid_cut=0.2):
    logger.debug(str(locals()))
    songs = []
    csvs = all_csv(directory)
    if len(csvs)==0:
        print "no csv files found in that directory"
        sys.exit()
    print 'loading these files:\n', csvs
    for csv in csvs:
        print "loading",csv
        data = load_csv(csv)
        ignore = ignore_track(data)
        chords = build_chords(data, ignore, noclip)
        songs.append(chords)
    songs ## has some length and chords
    l = len(songs)
    l1 = int(train_cut * l)
    l2 = int(valid_cut * l)
    data = {'train': songs[:l1], 'valid':songs[l1:l1+l2], 'test':songs[l1+l2:]}
    if save:
        with open(save,'wb') as file:
                cPickle.dump(data, file)
    return data


if __name__=="__main__":
    classPkl()
    '''    
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=str, help='specify the directory of the csv file outputs from midicsv')
    parser.add_argument('--save', required=True, type=str, help='specify what file to save the output to')
    parser.add_argument('--noclip', default=False, action='store_true', help='use this flag to not clip the range of notes in chords to 88 range of chord2vec')
    parser.add_argument('--train_cut', type=float, default=0.6, help='fraction of data to be used for training')
    parser.add_argument('--valid_cut', type=float, default=0.2, help='fraction of data to be used for validation')
    args = parser.parse_args(sys.argv[1:])
    
    main(**args.__dict__)
    '''
