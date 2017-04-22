import sys
import cPickle
import argparse

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

def build_chords(data,ignore_tracks,noclip):
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
            c_note = int(mes[4].strip())
            c_vel = int(mes[5].strip())
        except:
            continue
        if c_track not in ignore_tracks:
            if c_type=='Note_on_c' and int(c_vel)!=0 and c_note>=clip_min and c_note<=clip_max:
                if c_tic==p_tic:
                    current_chord.append(c_note)
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




def main(csv, save_file=None, clip=True):
    data = load_csv(csv)
    ignore = ignore_track(data)
    chords = build_chords(data,ignore,clip)
    if save_file:
        with open(save_file,'wb') as file:
                cPickle.dump(chords, file)
    return chords


if __name__=="__main__":
    noclip=False
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, type=str, help='specify the location of the csv file output from midicsv')
    parser.add_argument('--save', required=True, type=str, help='specify what file to save the output to')
    parser.add_argument('--noclip', action='store_true', help='use this flag to not clip the range of notes in chords to 88 range of chord2vec')
    
    args=parser.parse_args(sys.argv[1:])
    
    main(args.csv,args.save,noclip)
