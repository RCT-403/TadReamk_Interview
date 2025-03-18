import pandas as pd
import numpy as np
import nltk
from nltk.corpus import cmudict
import Levenshtein
from math import sqrt

designs = pd.read_csv('raw_designs.csv')
colors = pd.read_csv('raw_colors.csv')
corr = pd.read_csv('raw_correlation.csv')
titles = []
try:
    with open('final_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            titles.append(line.strip())
except UnicodeDecodeError as e:
    print(f"Error reading words.txt: {e}")


def weighted_jaccard_similarity(x, y):
    x_nonzero = x[x != 0].index
    y_nonzero = y[y != 0].index
    xy_nonzero = x_nonzero.union(y_nonzero)
    intersection = x_nonzero.intersection(y_nonzero)
    
    if len(intersection) == 0:
        if len(xy_nonzero) == 0:
            return 1
        return 0
    
    weights = []
    
    # For each column in xy_nonzero, get the max between all its previous entries xy_nonzero
    for i in range(len(xy_nonzero)):
        weight = 1
        max_cor = 0
        for j in range(i):
            if corr.iloc[int(xy_nonzero[i]), int(xy_nonzero[j])] > max_cor:
                weight = 1 - corr.iloc[int(xy_nonzero[i]), int(xy_nonzero[j])]
        weight = float(weight)
        weights.append(weight)
    # make a dictionary where each non_zero column is a key and the value is the weight
    weights_dict = dict(zip(xy_nonzero, weights))
    
    # calculate the weighted jaccard similarity
    numerator = 0
    denominator = 0
    for i in xy_nonzero:
        numerator += min(x[i], y[i]) * weights_dict[i]
        denominator += max(x[i], y[i]) * weights_dict[i]  
    
    if numerator == 0 and denominator == 0:
        return 1
    
    return numerator/denominator

def design_code_similarity_score(x,y):
    x = x.map(lambda i: 1 if i != 0 else 0)
    y = y.map(lambda i: 1 if i != 0 else 0)
    
    initial_score = weighted_jaccard_similarity(x, y)
    score = sqrt(initial_score)
    return score

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def color_similarity_score(x,y):
    cos_sim = cosine_similarity(x, y)
    score = cos_sim*cos_sim
    return score

d = cmudict.dict()

def word_to_phonetics(word):
    try:
        phonetic_representation = d[word.lower()][0]
        return phonetic_representation
    except KeyError:
        return None
    
def sentence_to_phonetics(sentence):
    words = sentence.split()
    phonetics = []
    for word in words:
        phonetic_representation = word_to_phonetics(word)
        if phonetic_representation:
            for phoneme in phonetic_representation:
                phonetics.append(phoneme)
        else:
            phonetics.append(word)
    return phonetics

def phonetic_similarity(phonetics1, phonetics2):
    # Join the phonetic representations into strings
    phonetic_str1 = ' '.join(phonetics1)
    phonetic_str2 = ' '.join(phonetics2)
    distance = Levenshtein.distance(phonetic_str1, phonetic_str2)
    max_len = max(len(phonetic_str1), len(phonetic_str2))
    if max_len == 0:
        return 1
    similarity = 1 - (distance / max_len)
    return similarity

def word_similarity_score(x,y):
    phonetics1 = sentence_to_phonetics(x)
    phonetics2 = sentence_to_phonetics(y)
    score = phonetic_similarity(phonetics1, phonetics2)
    score = sqrt(score)
    return score

def overall_similarity_score(x,y):
    design_code_score = design_code_similarity_score(designs.iloc[x], designs.iloc[y])
    color_score = color_similarity_score(colors.iloc[x], colors.iloc[y])
    title_score = word_similarity_score(titles[x], titles[y])
    weights = [4,1,3]
    score = (design_code_score*weights[0] + color_score*weights[1] + title_score*weights[2]) / sum(weights)
    return score





