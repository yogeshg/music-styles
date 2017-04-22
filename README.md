# music-styles
install midicsv: 
1) download souce code http://www.fourmilab.ch/webtools/midicsv/midicsv-1.1.tar.gz
2) extract ziped file
3) run make within extracted directory

convert midi to chords:
1) install midicsv above
2) in shell run: midicsv [midi file path] > [desired csv filepath]
3) run csv2chord.py --csv [csv filepath] --save [saved pkl file] 

This is a repository for a research project on music styles
repo wiki is found here: https://github.com/yogeshg/music-styles/wiki
