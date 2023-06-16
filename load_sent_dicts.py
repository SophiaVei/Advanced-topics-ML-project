#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def load_dicts(resources_folder='resources/'):
    print('Loading resources (lexicons + word vectors)...\n>Lexicons:')
    kbl_uni = read_tsv_resource(resources_folder+'KBL.tsv')
    kbl_bi = read_tsv_resource(resources_folder+'KBL.tsv', ngrams=2)
    print('\n\tKBL:', len(kbl_uni), 'unigrams +', len(kbl_bi), 'bigrams',
          '\n>Word vectors:')
    embeddings = read_csv_resource(resources_folder+'word2vec_greek.csv', 'e')
    print('\tword2vec:', len(embeddings), '\nDone!')
    return kbl_uni, embeddings, kbl_bi
    

def read_tsv_resource(resource_file, ngrams=1):
    resource = dict()
    words_to_num = read_file(resource_file)
    for word_to_num in words_to_num[1:len(words_to_num)]:
        row = word_to_num.split('\t')
        words = row[1].split()
        if len(words)==ngrams:
            resource[row[1]] = float(row[2])
    return resource


def read_csv_resource(resource_file, resource_type='grafs', ngrams=1):
    resource = dict()
    words_to_num = read_file(resource_file)
    start_from = 1 if resource_type=='grafs' else 0
    for word_to_num in words_to_num[start_from:len(words_to_num)]:
        row = word_to_num.split(',')
        if resource_type!='grafs':
            resource[row[0]] = [float(row[i]) for i in range(1, len(row))]
        elif len(row[0].split())==ngrams:
            resource[row[0]] = [float(row[i]) for i in range(1, len(row))]
    return resource
    
    
def read_file(infile):
    with open(infile, 'r', encoding='utf-8') as f:
        tweets = f.readlines()
    f.close()
    return tweets


'''kbl_uni, embeddings, kbl_bi = load_dicts()
print(len(kbl_bi))'''
