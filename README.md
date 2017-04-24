# music-styles
install midicsv: 
1) download souce code http://www.fourmilab.ch/webtools/midicsv/midicsv-1.1.tar.gz
2) extract ziped file
3) run make within extracted directory

convert midi to chords:
1) install midicsv above
2) in repo root create directory 'data'
3) cd into data
4) wget l_data http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz and unzip
5) wget http://ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-genreAssignment.cls
6) wget http://ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-partition_percentageSplit_0.8-v1.0.cls
7) create dir 'data/tmp'
8) run csv2chord.py 

This is a repository for a research project on music styles
repo wiki is found here: https://github.com/yogeshg/music-styles/wiki



MSD genre labels used: (stratified 80%train)
http://www.ifs.tuwien.ac.at/mir/msd/partitions/msd-topMAGD-partition_stratifiedPercentageSplit_0.8-v1.0.cls
