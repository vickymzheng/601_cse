#!/usr/bin/python

from operator import itemgetter
import sys

current_count = 0

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    current_count+=1

# do not forget to output the last word if needed!
print(current_count)
