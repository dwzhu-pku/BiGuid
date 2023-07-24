import os
import sys
os.chdir(sys.argv[1])

PATHLST = ["hypo","target"]

def is_chinese(uchar):

    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

for path in PATHLST:

    INFILE = f"./{path}"
    OUTFILE = f"./{path}.char"

    inLines = []
    outLines = []

    with open(INFILE,'r',encoding="utf-8") as fin:
        inLines = fin.readlines()

    for line in inLines:
        wdLst = list(line.replace("<<unk>>","unk").replace("<unk>","unk").replace("。 ", "。").strip())
        outLst = []
        cur = ""
        for char in wdLst :
            if cur == "":
                cur = cur + char
            elif is_chinese(cur) or is_chinese(char):
                outLst.append(cur)
                cur = char
            elif cur.isdigit() and char.isdigit():
                cur = cur + char
            elif cur.isalpha() and char.isalpha():
                cur = cur + char 
            else:
                outLst.append(cur)
                cur = char
        if cur != "":
            outLst.append(cur)
        outLines.append(" ".join(outLst))

    with open(OUTFILE,'w',encoding="utf-8") as fout:
        fout.writelines([f"{line}\n" for line in outLines])
    
    print(path,"zhMergeChar finished!")
