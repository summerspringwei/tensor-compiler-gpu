
from os import read


def read_lines(path):
    f = open(path, 'r')
    lines = f.readlines()
    s = set()
    for line in lines:
        s.add(line)
    f.close()
    return s

def compare():
    p1 = "o.txt"
    p2 = "t.txt"
    s1 = read_lines(p1)
    s2 = read_lines(p2)
    for a in s1:
        if a not in s2:
            print(a)

if __name__=="__main__":
    compare()