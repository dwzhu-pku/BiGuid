import os
import sys
os.chdir(sys.argv[1])

TGT = "./target.char"
SRC = "./hypo.char"

wdSet = set()
wdDict = dict()

with open(SRC, 'r', encoding="utf-8") as fin1, open(TGT, 'r', encoding="utf-8") as fin2:
    wdSet = set(fin1.read().replace("\n", " ").split(" "))
    wdSet.update(set(fin2.read().replace("\n", " ").split(" ")))

for idx, wd in enumerate(wdSet):
    wdDict[wd] = f"word{idx+1}"

wdDict["。"] = "."

print(list(wdDict)[:10])


def mapping(src):
    with open(src, 'r', encoding="utf-8") as fin1:
        lines1 = fin1.readlines()
        # print(lines1[:2])
        wtLines = []
        for line in lines1:
            wtLine = [str(wdDict.get(wd, 0)) for wd in line.strip().split(" ")]
            wtLines.append(" ".join(wtLine))
        # print(wtLines[:2])
    with open(f"{src}.en", 'w', encoding="utf-8") as fout1:
        for line in wtLines:
            fout1.write(line+"\n")


mapping(SRC)
mapping(TGT)
