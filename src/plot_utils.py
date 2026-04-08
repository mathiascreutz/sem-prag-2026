import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from IPython.display import display, Math, Latex
from scipy.cluster.hierarchy import dendrogram

def angle(vec1, vec2):
    """ Compute the angle in degrees between two vectors
        (The vectors are represented as lists or tuples that mark the
        end points; it is assumed that the starting point is at the
        origin)
    """
    unit1 = vec1 / np.linalg.norm(vec1)
    unit2 = vec2 / np.linalg.norm(vec2)
    
    dot_product = max(-1.0, min(1.0, np.dot(unit1, unit2)))
    
    return np.arccos(dot_product) * 180.0 / np.pi

def get_vector_color(i):
    """ An ordered range of colors used for the vectors """

    colors = [ "black", "blue", "red", "green",
               "magenta", "cyan", "brown", "yellow" ]
    
    return colors[i]

def print_angles_between_vectors(end_coords):
    """ Print all pairwise angles between a set of vectors """
    for i in range(len(end_coords)):
        c1 = get_vector_color(i).upper()
        for j in range(i + 1, len(end_coords)):
            c2 = get_vector_color(j).upper()
            print("The angle between the {:s} and {:s} vector is {:.1f} degrees."
                      .format(c1, c2, angle(end_coords[i], end_coords[j])))

def plot_vectors_2d(end_coords):
    """ Plot vectors in a 2d plane and count the angles between the vectors """

    if len(end_coords) < 1 or len(end_coords) > 8:
        print("Error: You need to plot at least one vector and at most eight vectors.")
        return
    
    fig = plt.figure()
    
    # Scale axes
    plt.axis([min(0, min(end_x for end_x, _ in end_coords))-1,
              max(0, max(end_x for end_x, _ in end_coords))+1,
              min(0, min(end_y for _, end_y in end_coords))-1,
              max(0, max(end_y for _, end_y in end_coords))+1])
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.grid(linestyle=":")
    
    for i, (end_x, end_y) in enumerate(end_coords):
        plt.arrow(0, 0, end_x, end_y,
                  head_width=0.2, 
                  length_includes_head=True,
                  color=get_vector_color(i))
        
    plt.show()
    
    print_angles_between_vectors(end_coords)

def plot_vectors_3d(end_coords):
    """ Plot vectors in a 3d plane and count the angles between the vectors """

    if len(end_coords) < 1 or len(end_coords) > 8:
        print("Error: You need to plot at least one vector and at most eight vectors.")
        return
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scale axes
    ax.set_xlim3d(min(0, min(end_x for end_x, _, _ in end_coords))-1,
                  max(0, max(end_x for end_x, _, _ in end_coords))+1)
    ax.set_ylim3d(min(0, min(end_y for _, end_y, _ in end_coords))-1,
                  max(0, max(end_y for _, end_y, _ in end_coords))+1)
    ax.set_zlim3d(min(0, min(end_z for _, _, end_z in end_coords))-1,
                  max(0, max(end_z for _, _, end_z in end_coords))+1)
    
    # Label axes
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    for i, (end_x, end_y, end_z) in enumerate(end_coords):
        ax.quiver(0, 0, 0, end_x, end_y, end_z,
                  arrow_length_ratio=0.2,
                  color=get_vector_color(i))
        
    plt.show()
    
    print_angles_between_vectors(end_coords)

def plot_3d_binary(features, word_feature_tuples, from_zero=False):
    """For plotting words with three binary features."""
    x = [coord[0] for _, coord in word_feature_tuples]
    y = [coord[1] for _, coord in word_feature_tuples]
    z = [coord[2] for _, coord in word_feature_tuples]

    fig = plt.figure()
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    if from_zero:
        # Set ticks
        ax.xaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
        ax.yaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
        ax.zaxis.set_ticks(list(map(lambda x: x/10.0, range(0, 15, 5))))
    else:
        ax.xaxis.set_ticks(range(-1, 2, 1))
        ax.yaxis.set_ticks(range(-1, 2, 1))
        ax.zaxis.set_ticks(range(-1, 2, 1))

    # Set label names
    ax.set_xlabel(features["x"])
    ax.set_ylabel(features["y"])
    ax.set_zlabel(features["z"])

    # Plot points
    ax.plot(x, y, z, marker="o", markersize=0, linestyle="None")
    # Plot arrows
    [ax.quiver(
        0, 0, 0, coord[0], coord[1], coord[2], length=1.0, arrow_length_ratio=0.1
     ) for _, coord in word_feature_tuples]
    
    # Plot words
    [ax.text(coord[0], coord[1], coord[2], word) for word, coord in word_feature_tuples]
    
    plt.show()
    
    
def get_principal_comps(M, n_components):
    """Get 'n' principal components of the matrix M.
    
    M is a matrix with dimensions (vocabulary, features).
    """
    if n_components not in [2, 3]:
        raise ValueError("Argument 'n_components' must be 2 or 3.")
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(M)

    comp_1 = pca_result[:,0]
    comp_2 = pca_result[:,1] 
    
    if n_components == 3:
        comp_3 = pca_result[:,2]
        return comp_1, comp_2, comp_3
    else:
        return comp_1, comp_2
        
        
def features_to_matrix(word_feature_tuples):
    """Convert Python list features to a numpy matrix.
    
    'word_feature_tuples' is a Python list of (word, feat_list) tuples.
    """
    feat_arrays = [np.array(feats).reshape(1, -1) for _, feats in word_feature_tuples]
    M = np.concatenate(feat_arrays)
    return M


def arrow(end_x, end_y):
    """Plot an arrow in 2d from (0, 0) to (end_x, end_y)."""
    plt.arrow(
        0, 0, 
        end_x, end_y, 
        head_width=0.1, 
        length_includes_head=True
    )
    
    
def arrow_3d(ax, end_x, end_y, end_z, color="blue", label=""):
    """Plot an arrow in 3d from (0, 0, 0) to (end_x, end_y, end_z).
    
    'ax' is an axis object from matplotlib.
    """
    ax.quiver(
        0, 0, 0,
        end_x, end_y, end_z, 
        arrow_length_ratio=0.1,
        color=color,
        label=label
    )

    
def point_label(word, x, y):
    """Plot label (i.e. word) to (x, y)."""
    plt.text(x, y, word, horizontalalignment='center', verticalalignment='bottom')
    
    
def point_label_3d(ax, word, x, y, z):
    """Plot label (i.e. word) to (x, y, z).
    
    'ax' is an axis object from matplotlib.
    """
    ax.text(x, y, z, word)
    
    
def plot_2d_binary_hd(word_feature_tuples, arrows=True):
    """Visualize high-dimensional embeddings in 2d space.
    
    'word_feature_tuples* is a Python list of (word, feat_list) tuples.
    """
    if arrows:
        msize = 0
    else:
        msize = 5
        
    mat = features_to_matrix(word_feature_tuples)
    pc_1, pc_2 = get_principal_comps(mat, 2)
    
    fig = plt.figure()
    plt.plot(pc_1, pc_2, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [arrow(pc_1[i], pc_2[i]) for i in range(len(word_feature_tuples))]
    
    [point_label(word, pc_1[i], pc_2[i]) for i, (word, _) in enumerate(word_feature_tuples)]
    plt.show()
    
    
def plot_3d_binary_hd(word_feature_tuples, arrows=True):
    """Visualize high-dimensional embeddings in 3d space.
    
    'word_feature_tuples' is a Python list of (word, feat_list) tuples.
    """
    if arrows:
        msize = 0
    else:
        msize = 5
    mat = features_to_matrix(word_feature_tuples)
    pc_1, pc_2, pc_3 = get_principal_comps(mat, 3)
    
    fig = plt.figure()
    # 3D plot
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pc_1, pc_2, pc_3, marker="o", markersize=msize, linestyle="None")
    if arrows:
        [arrow_3d(ax, pc_1[i], pc_2[i], pc_3[i]) for i in range(len(word_feature_tuples))]
        
    [point_label_3d(ax, word, pc_1[i], pc_2[i], pc_3[i]) for i, (word, _) in enumerate(word_feature_tuples)]
    
    
def plot_w2v_2d(words, embeddings, mapping, arrows=True):
    """Plot word2vec embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    pc_1, pc_2 = get_principal_comps(embeddings, 2)
    
    if arrows:
        msize = 0
    else:
        msize = 5
        
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    
    plt.figure()
    plt.plot(xs, ys, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [plt.arrow(
            0, 0, 
            xs[i], ys[i], 
            head_width=0.01, 
            length_includes_head=True
        ) for i in range(len(xs))]
    
    [point_label(word, xs[i], ys[i]) for i, word in enumerate(words)]
        
    plt.show()
    
def plot_w2v_3d(words, embeddings, mapping, arrows=True):
    """Plot word2vec embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    pc_1, pc_2, pc_3 = get_principal_comps(embeddings, 3)
    
    if arrows:
        msize = 0
    else:
        msize = 5
    words = [w for w in words if w in mapping]
    indices = [mapping[w] for w in words]
    
    xs = [pc_1[i] for i in indices]
    ys = [pc_2[i] for i in indices]
    zs = [pc_3[i] for i in indices]
    
    fig = plt.figure()
        
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, marker="o", markersize=msize, linestyle="None")
    
    if arrows:
        [arrow_3d(ax, xs[i], ys[i], zs[i]) for i in range(len(words))]
    
    [point_label_3d(ax, word, xs[i], ys[i], zs[i]) for i, word in enumerate(words)]
    
    plt.show()
    

def plot_sentences_3d(sentences, embeddings, mapping, embedding_fn=None):
    """Plot simple averaged sentence embeddings in 3d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    if not embedding_fn:
        embedding_fn = embed_sentence
        
    embs_sent = [embedding_fn(s, embeddings, mapping) for s in sentences]
    M_sent = np.concatenate(embs_sent)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(M_sent)

    comp_1_sent = pca_result[:,0]
    comp_2_sent = pca_result[:,1] 
    comp_3_sent = pca_result[:,2] 

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    [ax.scatter(comp_1_sent[i], comp_2_sent[i], comp_3_sent[i]) for i, _ in enumerate(comp_1_sent)]
    [ax.plot([0, comp_1_sent[i]], [0, comp_2_sent[i]], [0, comp_3_sent[i]]) for i, _ in enumerate(comp_1_sent)]

    legend_texts = [s[:20] + "..." for s in sentences]
    plt.legend(legend_texts, bbox_to_anchor=(0.6,0.6))
    plt.show()
    
    
def plot_sentences_2d(sentences, embeddings, mapping, embedding_fn=None):
    """Plot simple averaged sentence embeddings in 2d.
    
    'embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    if not embedding_fn:
        embedding_fn = embed_sentence
        
    embs_sent = [embedding_fn(s, embeddings, mapping) for s in sentences]
    M_sent = np.concatenate(embs_sent)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(M_sent)

    comp_1_sent = pca_result[:,0]
    comp_2_sent = pca_result[:,1] 

    plt.figure()
    [plt.scatter(comp_1_sent[i], comp_2_sent[i]) for i, _ in enumerate(comp_1_sent)]
    [plt.plot([0, comp_1_sent[i]], [0, comp_2_sent[i]]) for i, _ in enumerate(comp_1_sent)]
    
    legend_texts = [s[:20] + "..." for s in sentences]
    plt.legend(legend_texts, bbox_to_anchor=(0.6,0.6))
    
    plt.show()

def rank_sentences_by_similarity(tgt, sents, embeddings,
                                 mapping, embedding_fn=None):    
    """ Sort sentences by similarity compared to a given
    target sentence.
    """
    if not embedding_fn:
        embedding_fn = embed_sentence

    emb_tgt = embedding_fn(tgt, embeddings, mapping).reshape(-1)
    embs_sents = [ embedding_fn(s, embeddings, mapping).reshape(-1) for s in sents ]
    angles = [ angle(emb_tgt, e) for e in embs_sents ]

    sorted_sents = sorted(zip(sents, angles), key=lambda item: item[1])

    print('Target sentence: "{:s}"\n'.format(tgt))
    print("Similarity to other sentences (measured as angle in degrees):")
    for i, (s, a) in enumerate(sorted_sents):
        print("#{:d} {:s}: {:.1f}".format(i, s, a))
    
def get_embeddings():
    """Load pretrained embeddings and word to int mappings."""
    with open("../../../shared/embedding-matrix-en.npy", "rb") as f:
        M = np.load(f)
    with open("../../../shared/wtoi-en.pickle", "rb") as f:
        wtoi = pickle.load(f)
        
    return M, wtoi

def embed_sentence(sentence, word_embeddings, mapping):
    """Embed sentence using simple word averaging method.
   
    'sentence' is an all-lowercase string with tokens separated by spaces.
   
    'word_embeddings' contains the word embeddings in a numpy matrix
    and 'mapping' is a {str:int} dict mapping words to row indices
    in 'embeddings'.
    """
    # Get the row indices of the words in the sentence
    indices = [mapping[w] for w in sentence.split() if w in mapping]
    
    # You can index a numpy array by giving it a Python list of integers.
    # word_vectors is now a (|s| x 300) array (matrix), where |s| is the
    # length of the sentence. (Actually this only holds if all the words
    # in the sentence have embeddings, so in reality it is probably not
    # |s|.)
    word_vectors = word_embeddings[indices]
    
    # Convenient averaging function, notice the correct axis (which would
    # probably be correct by default, but anyways..). "np.average" is used
    # here instead of "np.mean" because "average" could also be weighted.
    sentence_embedding = np.average(word_vectors, axis=0)
    
    # Reshape from 1d array to (1 x 300) for convenience.
    return sentence_embedding.reshape(1, -1)


def arrow_from(start_x, start_y, end_x, end_y, c="black", linestyle='solid'):
    """Plot an arrow in 2d from (start_x, start_y) to (end_x, end_y)."""
    a = plt.arrow(
        start_x, start_y, 
        end_x, end_y,
        width=0.001,
        head_width=0.02, 
        length_includes_head=True,
        color=c,
        linestyle=linestyle
    )
    return a

def plot_w2v_algebra(embeddings, mapping, base, minus=None, plus=None, results=[]):
    """Plot vector algebra of the form 'base - minus + plus = ?'"""

    if not base:
        print("Error: You need to give a base word.")
        return

    if not base in mapping:
        print("Error: out of vocabulary word:", base)
        return

    if minus and not minus in mapping:
        print("Error: out of vocabulary word:", minus)
        return

    if plus and not plus in mapping:
        print("Error: out of vocabulary word:", plus)
        return

    if results:
        for result in results:
            if result not in mapping:
                print("Error: out of vocabulary word:", result)
                return

    words = [w for w in [base, minus, plus] if w]
    if results:
        words.extend(results)

    indices = [mapping[w] for w in words]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    comp_1 = pca_result[:,0][indices]
    comp_2 = pca_result[:,1][indices]
    
    wtoi = {w:i for i, w in enumerate(words)}
    itow = {i:w for i, w in enumerate(words)}
    
    base_i = wtoi[base]
    if minus:
        minus_i = wtoi[minus]
    if plus:
        plus_i = wtoi[plus]
    
    # The vectors to be plotted
    if minus and plus:
        base_minus_x = comp_1[base_i] - comp_1[minus_i]
        base_minus_y = comp_2[base_i] - comp_2[minus_i]

        final_x = base_minus_x + comp_1[plus_i]
        final_y = base_minus_y + comp_2[plus_i]

    elif minus:
        final_x = comp_1[base_i] - comp_1[minus_i]
        final_y = comp_2[base_i] - comp_2[minus_i]

    elif plus:
        final_x = comp_1[base_i] + comp_1[plus_i]
        final_y = comp_2[base_i] + comp_2[plus_i]
    
    # Plot dots at end of vectors (and origin and result)
    all_x = np.concatenate([[0], [comp_1[base_i]]])
    all_y = np.concatenate([[0], [comp_2[base_i]]])
    
    if minus and plus:
        all_x = np.concatenate([all_x, [base_minus_x]])
        all_y = np.concatenate([all_y, [base_minus_y]])
        
    if minus or plus:
        all_x = np.concatenate([all_x, [final_x]])
        all_y = np.concatenate([all_y, [final_y]])

    if results:
        for result in results:
            equals = wtoi[result]
            all_x = np.concatenate([all_x, [comp_1[equals]]])
            all_y = np.concatenate([all_y, [comp_2[equals]]])
    
    plt.figure()
    plt.plot(
        all_x,
        all_y, 
        marker="o", 
        markersize=2.0, 
        linestyle="None"
    )

    # Plot vectors
    arr1 = arrow_from(0, 0, comp_1[base_i], comp_2[base_i])
    legend_arrows = [ arr1 ]
    legend_labels = [ base ]

    if minus:
        arr2 = arrow_from(
            comp_1[base_i], 
            comp_2[base_i], 
            -1*comp_1[minus_i], 
            -1*comp_2[minus_i], 
            "red"
        )
        legend_arrows.append(arr2)
        legend_labels.append(" – " + minus)

    if minus and plus:
        arr3 = arrow_from(
            base_minus_x, 
            base_minus_y, 
            comp_1[plus_i], 
            comp_2[plus_i], 
            "blue"
        )
        legend_arrows.append(arr3)
        legend_labels.append(" + " + plus)

    elif plus:
        arr3 = arrow_from(
            comp_1[base_i], 
            comp_2[base_i], 
            comp_1[plus_i], 
            comp_2[plus_i], 
            "blue"
        )
        legend_arrows.append(arr3)
        legend_labels.append(" + " + plus)

    # Origin to final
    if minus or plus:
        arr4 = arrow_from(0, 0, final_x, final_y, "orange", linestyle="--")
        legend_arrows.append(arr4)
        if minus and plus:
            legend_labels.append(base + " – " + minus + " + " + plus)
        elif minus:
            legend_labels.append(base + " – " + minus)
        else:
            legend_labels.append(base + " + " + plus)

    if results:
        for result in results:
            equals = wtoi[result]
            plt.text(comp_1[equals], comp_2[equals], result)

    # Plot legend
    plt.legend(legend_arrows, legend_labels, fancybox=True)

    plt.show()

def find_true_closest(embeddings, mapping, base, minus, plus, n=5):
    true_final = embeddings[mapping[base]] - embeddings[mapping[minus]] + embeddings[mapping[plus]]
     
    distances = cosine_distances(true_final.reshape(1, -1), embeddings).reshape(-1)
    closest_indices = distances.argsort()[-n:][::-1] 
     
    return closest_indices

def to_dict(l):
    return {k:v for k, v in l}

def tabulate_angles(words_and_feats):
    word_dict = to_dict(words_and_feats)
    latex = "\\textbf{Pairwise angles} \\\\\n"
    latex += "\\begin{array}{" + "c|" + "c"*len(word_dict) + "}\n& "

    for word in word_dict.keys():
        latex += "\\textrm{%s} & " % word
    
    latex = latex[:-1] + r" \\\hline" + "\n"

    for word1 in word_dict.keys():
        latex += "\\textrm{%s}" % word1
        for word2 in word_dict.keys():
            latex += r" & %.1f°" % angle(word_dict[word1], word_dict[word2])
        latex += r" \\" + "\n"

    latex += "\n\\end{array}"
    display(Math(latex))

def plot_dendrogram(model, labels):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    
    labels_concat = ["/".join(tup) for tup in zip(labels, [str(l) for l in model.labels_])]

    # Plot the corresponding dendrogram
    plt.figure()

    dendrogram(linkage_matrix, orientation="left", labels=labels_concat)

    plt.show()

def embed(w, M, mapping):
    if w not in mapping:
        raise ValueError("Unfortunately '" + w + "' is not in the vocabulary.")
    return M[mapping[w]]

def to_feature_matrix(words, M, mapping):
    return np.concatenate([embed(w, M, mapping).reshape(1, -1) for w in words])

def plot_kmeans(model, words, embeddings, mapping, plot_text=True, small_points=False):
    clusters = max(model.labels_) + 1
    
    pc_1, pc_2 = get_principal_comps(embeddings, 2)
    
    fig = plt.figure()
    for c in range(clusters):
        tmp = [w for w, cluster_id in zip(words, model.labels_) if cluster_id == c]
        
        indices = [mapping[w] for w in tmp]
            
        xs = [pc_1[i] for i in indices]
        ys = [pc_2[i] for i in indices]
        
        if small_points:
            plt.plot(xs, ys, marker="o", markersize=0.5, linestyle="None", label="Cluster " + str(c))
        else:
            plt.plot(xs, ys, marker="o", linestyle="None", label="Cluster " + str(c))
    
        if plot_text:
            [point_label(tmp[i], xs[i], ys[i]) for i, word in enumerate(tmp)]
            
        if clusters < 14:
            if small_points:
                plt.legend(markerscale=7.)
            else:
                plt.legend()
        
        
    plt.show()

def sample_clusters(model, words, n, clusters):
    cluster_to_words = {c:[] for c in range(max(model.labels_) + 1)}
    [cluster_to_words[cluster_id].append(w) for w, cluster_id in zip(words, model.labels_)]
    
    for c in clusters:
        print("Cluster %d:" % c)
        print(", ".join(random.sample(cluster_to_words[c], min(n, len(cluster_to_words[c])))), end="\n\n")
        
def get_clusters_at_cutoff(model, words, cutoff):
    children = model.children_[:cutoff + 1]
    clusters = {i:[w] for i, w in enumerate(words)}
    
    tmp = len(words)
    for c1, c2 in children:
        if tmp not in clusters:
            clusters[tmp] = []
        
        for c in [c1, c2]:
            if c < len(words):
                clusters[tmp].append(words[c])
            else:
                clusters[tmp] += clusters[c]
        
        clusters[c1] = None
        clusters[c2] = None
        
        tmp +=1
        
    clusters = [w for k, w in clusters.items() if w]
    
    for i, c in enumerate(clusters):
        print("Cluster %d: %s" % (i, ", ".join(c)))
