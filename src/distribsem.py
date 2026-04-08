import matplotlib.pyplot as plt
import numpy as np
import plot_utils
import operator
import re

from mpl_toolkits.mplot3d import Axes3D
from nltk.probability import FreqDist

def filter_by_frequency(text, amt_to_keep, do_not_filter):
    fd = FreqDist(text)
    keep = [w for w, _ in fd.most_common(amt_to_keep)] + do_not_filter
    text = [w if w in keep else "<UNK>" for w in text]
    return text

def show_kwic(text, word, window, dimensionality, show_n):
    text = filter_by_frequency(text, dimensionality, [word])
    
    occurrences = 0
    word = word.lower()
    for i, token in enumerate(text):
        if token.lower() == word:
            left_context = " ".join(text[max(0, i - window):i])
            right_context = " ".join(text[i + 1:i + window + 1])
            print("{:>50}  {}  {:<50}".format(left_context, token, right_context))
            
            occurrences += 1
            if occurrences > show_n:
                break
            

def create_vectors(dimensionality, window_size, text, freq_thresh=15):

    text = [w.lower() for w in text]
    
    fd = FreqDist(text)

    vocabulary = [ w for w, f in sorted(fd.items(), key=operator.itemgetter(1), reverse=True) \
                   if f >= freq_thresh ]
    vocabtoi = { w:i for i, w in enumerate(vocabulary) }
    
    context_vocabulary = [ w for w, _ in fd.most_common(dimensionality) ]
    contexttoi = { w:i for i, w in enumerate(context_vocabulary) }

    M = np.ones((len(vocabulary), dimensionality))

    for i, token in enumerate(text):
        if token in vocabulary:
            context = text[max(0, i - window_size):i] + text[i + 1:i + window_size + 1]
            for ct in context:
                if ct in context_vocabulary:
                    M[vocabtoi[token], contexttoi[ct]] += 1
                    
    M = normalize_M(M)
    
    return M, vocabtoi

def create_vectors_shared(max_vocab_size, min_dimensionality, window_size, text1, text2):
    text1 = [w.lower() for w in text1]
    text2 = [w.lower() for w in text2]
    
    fd1 = FreqDist(text1)
    fd2 = FreqDist(text2)
    
    vocabulary1 = [w for w, _ in fd1.most_common(max_vocab_size)]
    vocabulary2 = [w for w, _ in fd2.most_common(max_vocab_size)]
    vocabulary = set(vocabulary1).intersection(set(vocabulary2))
    print("Final vocabulary size:", len(vocabulary))
    
    vocabtoi = {w:i for i, w in enumerate(vocabulary)}
    
    context_vocabulary1 = [w for w, _ in fd1.most_common(min_dimensionality)]
    context_vocabulary2 = [w for w, _ in fd2.most_common(min_dimensionality)]
    context_vocabulary = set(context_vocabulary1).union(set(context_vocabulary2))
    print("Final context vocabulary size (embedding dimensionality):", len(context_vocabulary))
    
    contexttoi = {w:i for i, w in enumerate(context_vocabulary)}
    
    M1 = np.ones((len(vocabulary), len(context_vocabulary)))
    M2 = np.ones((len(vocabulary), len(context_vocabulary)))
    
    for i, token in enumerate(text1):
        if token in vocabulary:
            context = text1[max(0, i - window_size):i] + text1[i + 1:i + window_size + 1]
            for ct in context:
                if ct in context_vocabulary:
                    M1[vocabtoi[token], contexttoi[ct]] += 1
                    
    for i, token in enumerate(text2):
        if token in vocabulary:
            context = text2[max(0, i - window_size):i] + text2[i + 1:i + window_size + 1]
            for ct in context:
                if ct in context_vocabulary:
                    M2[vocabtoi[token], contexttoi[ct]] += 1
                    
    M1 = normalize_M(M1)
    M2 = normalize_M(M2)
    
    return M1, M2, vocabtoi
    
def normalize_M(M):
    return M/np.sqrt(np.sum(np.power(M, 2), axis=1).reshape(-1, 1))
    
def plot_embeddings(words, embeddings, mapping, arrows=True):
    plot_utils.plot_w2v_3d(words, embeddings, mapping, arrows=False)
    
def plot_two_embeddings(words, embeddings_1, embeddings_2, 
                        mapping_1, mapping_2=None, 
                        embeddings_1_name="emb1", 
                        embeddings_2_name="emb2",
                        arrows=False):
    
    if not mapping_2:
        mapping_2 = mapping_1
    
    embeddings = np.concatenate([embeddings_1, embeddings_2], axis=0)
    pc_1, pc_2, pc_3 = plot_utils.get_principal_comps(embeddings, 3)
    
    pc_11, pc_21 = np.split(pc_1, 2)
    pc_12, pc_22 = np.split(pc_2, 2)
    pc_13, pc_23 = np.split(pc_3, 2)
    
    if arrows:
        msize = 0.001
    else:
        msize = 5
        
    words_not_in_mapping = []
    for w in words:
        if not (w in mapping_1 and w in mapping_2):
            words_not_in_mapping.append(w)
    if words_not_in_mapping:
        print("The following words do not have embeddings:", words_not_in_mapping)
        
    words = [w for w in words if w in mapping_1 and w in mapping_2]
    indices_1 = [mapping_1[w] for w in words]
    indices_2 = [mapping_2[w] for w in words]
    
    xs1 = [pc_11[i] for i in indices_1]
    ys1 = [pc_12[i] for i in indices_1]
    zs1 = [pc_13[i] for i in indices_1]
    
    xs2 = [pc_21[i] for i in indices_2]
    ys2 = [pc_22[i] for i in indices_2]
    zs2 = [pc_23[i] for i in indices_2]
    
    fig = plt.figure()
        
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim([-0.5, 0.9])
    ax.set_ylim([-0.5, 0.9])
    ax.set_zlim([-0.75, 0.75])
    
    ax.plot(
        xs1, ys1, zs1,
        marker="o", 
        markersize=msize, 
        linestyle="None", 
        label=embeddings_1_name
    )
    ax.plot(
        xs2, ys2, zs2, 
        marker="o", 
        markersize=msize, 
        linestyle="None", 
        color="orange",
        label=embeddings_2_name
    )
    
    if arrows:
        [plot_utils.arrow_3d(ax, xs1[i], ys1[i], zs1[i]) for i in range(len(words))]
        [plot_utils.arrow_3d(ax, xs2[i], ys2[i], zs2[i], color="orange") for i in range(len(words))]
    
    plt.legend()
    
    [plot_utils.point_label_3d(ax, word, xs1[i], ys1[i], zs1[i]) for i, word in enumerate(words)]
    [plot_utils.point_label_3d(ax, word, xs2[i], ys2[i], zs2[i]) for i, word in enumerate(words)]
    
    plt.show()
    
    
def filter_text(text):
    text = [re.sub("[^A-Za-z.!?:]", "", w) for w in text if re.sub("[^A-Za-z,.!?:]", "", w) != ""]
    return text
