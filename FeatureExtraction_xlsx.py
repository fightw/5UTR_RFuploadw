####extract feature for latter prediction use
###RNA folding
###Condon
### k-mer
### motif
import concurrent
import multiprocessing
import sys,os
from Bio import SeqIO
import Bio.SeqUtils.CodonUsage
import subprocess
from multiprocessing import Pool, cpu_count, freeze_support
import gzip
from main import *
import pandas as pd

cds_length=15 ##assumption last portion of sequence is cds

#inputFasta="output/test.fa"
# inputFasta = sys.argv[1]
out_dir = 'data' ##output directory


###take the longest length for the same transcript id
import csv


###take the longest length for the same transcript id

# print(seq)

import pandas as pd

# 从 Excel 文件中读取数据
df = pd.read_csv('ABIvsTEfeatureProject/PC3聚类版.csv', encoding='utf-8')

# df = pd.read_csv('data/traingene-5UTR-6720.csv')

tx_seq = dict()

for _, row in df.iterrows():
    tx = row['geneName']  # 转录本ID所在列的列名
    seq = row['seq']  # 序列所在列的列名
    print(seq)

    if len(seq) < 30:  # 当序列太短时跳过
        continue
    # if "ATG" not in seq:  # 当序列中不包含起始密码子时跳过
    #     continue
    if tx not in tx_seq or len(seq) > len(tx_seq[tx]):
        tx_seq[tx] = seq

# print(tx_seq)





###output the non-redundancy fasta
outputFasta=out_dir+"/input.filter.fa"
outf = open(outputFasta,"w")
txIDlist = list()
seqList = list()
for tx in tx_seq:
    outf.write(">"+str(tx)+"\n")
#    outf.write(tx_seq[tx].tostring()+"\n")
    outf.write(str(tx_seq[tx])+ "\n")
    txIDlist.append(tx)
    seqList.append(tx_seq[tx])
outf.close()

#GTGAGCGACACAGAGCGGGCCGCCACCGCCGAGCAGCCCTCCGGCAGTCTCCGCGTCCGTTAAGCCCGCGGGTCCTCCGCGAATCGGCGGTGGGTCCGGCAGCCGAATGCAGCCCCGCAGCG
# outf2.close()
if __name__ == '__main__':
    # seqStringList = [str(seq) for seq in seqList]
    # print(seqList)
    print('==============================================================================')
    # print(str(seqList))
    with Pool(multiprocessing.cpu_count()) as pool:
        featList = pool.map(extract_features,  seqList)
        # featList = pool.map(Seq2Feature,  'GCCCAGTTGGCTGGACCAATGGATGGAGAGAATC')

    outf2 = gzip.open(out_dir + "/input.fa.sparseFeature.txt.gz", 'wt')
    feat2ID = dict()
    featid = -1
    for i in range(len(txIDlist)):
        txid = i
        for featItem in featList[i]:
            featname = featItem[0]#特征名
            featVal = featItem[1]#特征值
            # print('featList' + featList[0] +featItem[1])
            if featname not in feat2ID:
                featid += 1
                feat2ID[featname] = featid
                fid = featid
            else:
                fid = feat2ID[featname]
            outstr = str(i) + "\t" + str(fid) + "\t" + str(featVal)
            outf2.write(outstr + "\n")

    outf2.close()

    ##mapping id to human understandable name
    outf3 = open(out_dir + "/input.fa.sparseFeature.rowname", 'w')
    for i in range(len(txIDlist)):
        outf3.write(str(i) + "\t" + str(txIDlist[i]) + "\n")

    outf3.close()

    outf4 = open(out_dir + "/input.fa.sparseFeature.colname", 'w')
    sorted_items = sorted(feat2ID.items(), key=lambda x: x[1])
    for a in sorted_items:
        print(a)
        featname = a[0].strip()
        fid = a[1]
        outf4.write(str(fid) + "\t" + str(featname) + "\n")
    outf4.close()
    pool.close()
    pool.join()





