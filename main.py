import subprocess
from collections import Counter
import RNA
from Bio.Seq import Seq
from Bio.Data import CodonTable
import re


def extract_kmer_features(sequence, max_k):
    # kmer_frequencies = {}
    feature_map = dict()
    for k in range(1, max_k + 1):
        kmer_counts = {}
        sequence_length = len(sequence)
        # 遍历序列，统计每个K-mer出现的次数
        for i in range(sequence_length - k + 1):
            kmer = sequence[i:i+k]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1
        # 将K-mer出现次数转换为频率
        total_kmers = sequence_length - k + 1
        feature_map[k] = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return feature_map

def extract_ngram_features(sequence, n):
    feature_map = dict()
    print(sequence)
    sequence = sequence.translate(str.maketrans("", ""))
    for i in range(len(sequence) - n + 1):
        ngram = sequence[i:i+n]
        if ngram in feature_map:
            feature_map[ngram] += 1
        else:
            feature_map[ngram] = 1
    return feature_map

def codonFreq(seq):
    # 清理序列，只保留有效的核苷酸字符
    seq = ''.join([base for base in seq if base in 'ACGT'])

    # 检查序列是否至少有三个核苷酸
    if len(seq) < 3:
        print(f"序列过短，无法翻译: {seq}")
        return None  # 可以根据需要返回默认值

    # 检查序列长度是否为3的倍数
    if len(seq) % 3 != 0:
        # 如果不是，进行裁剪或填充（例如裁剪掉最后不足3个核苷酸的部分）
        seq = seq[:len(seq) - len(seq) % 3]  # 裁剪到最近的3的倍数长度
    try:
        # 翻译序列
        codon_str = str(Seq(seq).translate())
    except CodonTable.TranslationError as e:
        print(f"翻译错误: {seq}: {e}")
        return None  # 或者根据需要返回默认值

    # 计算每种密码子的出现频率
    tot = len(codon_str)
    feature_map = dict()
    for a in codon_str:
        a = "codon_" + a
        if a not in feature_map:
            feature_map[a] = 0
        feature_map[a] += 1.0 / tot

    # 统计启动密码子 (M 表示启动密码子 'ATG' 对应的氨基酸)
    feature_map['uAUG'] = codon_str.count("M")  # 启动密码子 'ATG'
    # 统计终止密码子 (Stop codons: '*')
    feature_map['uORF'] = codon_str.count("*")  # 停止密码子
    return feature_map  # 返回特征字典

def extract_features(seq):
    feature_list = []
    seq=seq.upper()
    # 通过调用 extract_kmer_features 函数获取 K-mer 特征
    N_count = dict()  # add one pseudo count
    N_count['C'] = 1
    N_count['G'] = 1
    N_count['A'] = 1
    N_count['T'] = 1

    kmer_features = extract_kmer_features(seq, 6)
    for k, frequencies in kmer_features.items():
        for kmer, frequency in frequencies.items():
            feature_name = f"kmer_{kmer}"
            feature_value = frequency
            feature_list.append((feature_name, feature_value))

    seq_str = str(seq)  # 将 Seq 对象转换为字符串
    # 处理序列数据
    codon_features = codonFreq(seq_str)
    for feat_name, feat_value in codon_features.items():
        feature_list.append((feat_name, feat_value))

    for a in seq_str:
        if a not in N_count:
            N_count[a] = 0
        N_count[a] += 1

    CGperc=float(N_count['C']+N_count['G'])/len(seq_str)
    CGratio=abs(float(N_count['C'])/N_count['G']-1)
    ATratio=abs(float(N_count['A'])/N_count['T']-1)

    rna_seq = Seq(seq_str).transcribe()
    # Predict RNA secondary structure and calculate energy
    fold = RNA.fold(str(rna_seq))
    energy = RNA.energy_of_structure(str(rna_seq), fold[0], 0)
    # Calculate the mean base pair distance of RNA structure
    fc = RNA.fold_compound(str(rna_seq))
    fc.pf()
    mean_distance = fc.mean_bp_distance()

    md = RNA.md()
    md.gquad = 1
    fc = RNA.fold_compound(seq, md)

    # predict Minmum Free Energy and corresponding secondary structure
    (ss, mfe) = fc.mfe()
    gquad_energy = mfe # 1表示考虑G-四链体的结构
    feature_list.append(('energy', energy))
    # feature_list.append(('kozak', kozak))
    feature_list.append(('CGperc', CGperc))
    feature_list.append(('CGratio', CGratio))
    feature_list.append(('ATratio', ATratio))
    feature_list.append(('mean_distance', mean_distance))
    feature_list.append(('gquad_energy', gquad_energy))
    # feature_list.append(('last_30_bases_energy', last_30_bases_energy))
    # 通过调用 codonFreq 函数获取 Codon 特征
    print(feature_list)
    return feature_list


def Seq2Feature(dna_sequence):
    # Translate DNA sequence to protein sequence
    translated_seq = Seq(dna_sequence).translate()

    # Count codon frequency
    codon_freq = dict(Counter(translated_seq))

    # Extract 3-gram frequency
    ngram_features = extract_ngram_features(translated_seq, 3)

    # Transcribe DNA sequence to RNA sequence
    rna_seq = Seq(dna_sequence).transcribe()

    # Predict RNA secondary structure and calculate energy
    fold = RNA.fold(str(rna_seq))
    energy = RNA.energy_of_structure(str(rna_seq), fold[0], 0)

    # Calculate the mean base pair distance of RNA structure
    fc = RNA.fold_compound(str(rna_seq))
    fc.pf()
    mean_distance = fc.mean_bp_distance()


    # Calculate codon frequency and related statistics
    freq_dict = codonFreq(dna_sequence)

    # Combine all features into one list
    feature_list = []
    feature_list.append(('codon_' + k, v) for k, v in codon_freq.items())
    feature_list.append(('ngram_' + k, v) for k, v in ngram_features.items())
    # for k in range(1, max_k + 1):
    #     feature_list.append(('kmer_' + k + '_' + x, y) for x, y in kmer_features[k].items())
    feature_list.append(('energy', energy))
    feature_list.append(('mean_distance', mean_distance))
    feature_list.append(('codonFreq_' + k, v) for k, v in freq_dict.items())

    return feature_list

