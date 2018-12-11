# load the file
import os
import glob
import re
from collections import Counter
import ntpath
import sys
import json

path = 'data/'

cleanedPath = './cleaned/'
os.makedirs(cleanedPath, mode=0o777, exist_ok=True)

# gets all the files in the directory
dir_name = "./"
all_files = os.listdir(dir_name)

# removes all .txt files before starting to run the program
for item in all_files:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))

def path_leaf(x):
    head, tail = ntpath.split(x)
    return tail or ntpath.basename(head)

read = False

titles = {}

# # getting all files in single file for words count
# for filename in glob.glob(os.path.join(path, '*')):
#     titleList = []
#     with open(filename, 'r') as infile, open((cleanedPath+path_leaf(filename)), 'w') as outfile:
#         lines = infile.readlines()
#         for line in lines:
#             if read:
#                 outfile.write(line[2:-2] + "\n")
#                 titleList.append(line[2:-2])
#             if line == "sg6\n":
#                 read = True
#             else:
#                 read = False
#         titles[path_leaf(filename)[:-4]] = titleList
#
# infile.close()
# outfile.close()
#
# # for k,v in titles.items():
# #     print(k, len(v))
#
# with open("titles.txt", 'w') as dictFile:
#     json.dump(titles, dictFile)
#
#
# print(len(titles))
