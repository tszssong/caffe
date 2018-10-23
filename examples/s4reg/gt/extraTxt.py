import sys
import os

extractfile = sys.argv[1]
gtfilepath = sys.argv[2]

extractlist = []
for line in open(extractfile):
    filename = line.strip().split("/")[-1]
#    print "extra:", filename
    extractlist.append(filename)
#print extractlist

gtlist = []
gtdict = {}
for line in open(gtfilepath):
    filename = line.strip().split(".")[0]
    print "gt:",filename
    gtlist.append(filename)
    gtdict[filename] = line
#print gtlist

f = open('1in2'+sys.argv[1], 'w')
for filename in extractlist:
    if gtdict.has_key(filename):
        f.write(gtdict[filename])
    else:
        print "not exit in gt", filename
f.close()


