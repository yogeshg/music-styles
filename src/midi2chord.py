import sys

def load_csv(filename):
    import csv
    data=[]
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
            print ', '.join(row)
    return data

def ignore_track(data):
    ignore=[]
    for i in range(len(data)):
        if data[i][2]=="Program_c" and data[i][4] <= 88:
            ignore.append(data[i][0])
    return ignore

def build_chords(data,ignore_tracks):
    sorted_data = sorted(data, key=lambda x: int(x[1]))
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
            if c_type=='Note_on_c' and int(c_vel)!=0:
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
    print chords

    return chords




def main(filename):
    data = load_csv("1.csv")
    ignore = ignore_track(data)
    chords = build_chords(data,ignore)
    return chords


if __name__=="__main__":
    
    main("1.csv")
