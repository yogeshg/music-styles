#!/bin/bash -x

#rm -r models/ results/
#for f in *.tar.gz; do gtar -xvf $f; done
cd models/archive
for f in *.tar; do p=${f%.*};  gtar --xform="s/.*\/\(.*\)/${p}_\1/" -xvf $f; done
rm -r current_*
rm *.tar
cd ../..
cd results/archive
for f in *.tar; do p=${f%.*}; gtar --xform="s/.*\/\(.*\)/${p}_\1/" -xvf $f; done
rm -r current_*
rm *.tar
cd ../..
