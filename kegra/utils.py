from __future__ import print_function

import os.path
import mmap
from nltk.corpus import brown as bro

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot    #onehot googlen


def contentbuilder(path="data/partofspeech/glove.6B/", dataset="glove.6B", small_dataset = False):
    
    # check if file already exists otherwise create and write
    #===========================================================================
    # try:
    #     file = open("/data/brown/brown.content", 'r')
    # except FileNotFoundError:
    #     file = open("/data/brown/brown.content", 'w')
    #===========================================================================
    
    content = "data/brown/brown.content"
    cites = "data/brown/brown.cites"
    ids = "data/brown/brown.ids"
    if not os.path.exists(content) and not os.path.exists(cites):
        os.makedirs(os.path.dirname(content))
         
    #if os.path.isfile("/data/brown/brown.content") and os.path.isfile("/data/brown/brown.cites"):
    #    return
    
    print('Building {} content...'.format(dataset))
    
    content_file = open(content, 'w')
    
    cites_file = open(cites, 'w')
    
    ids_file = open(ids, 'w')
    
    #mmap_content = mmap.mmap(content_file.fileno(), 0, access=mmap.ACCESS_READ)
    #mmap_cites = mmap.mmap(cites_file.fileno(), 0, access=mmap.ACCESS_READ)
    
    glove_dict = {}
    with open("{}{}.50d.txt".format(path, dataset), encoding="utf8") as glove_file:
        for line in glove_file:
            line_split = line.split()
            glove_dict[line_split[0]] = line_split[1:]
            
    glove_file.close()
    
    # generate brown tagged words list, not necessary but just in case
    #===========================================================================
    # with open("data/brown/brown.tags", 'w') as tags_file:
    #     tags_file.write('\n'.join('%s %s' % x for x in bro.tagged_words()))
    #===========================================================================
    
    
    brown_tags = bro.tagged_words()
    
    if small_dataset == False:
        id_curr = 0
        id_prev = 0
        for wordtuple in brown_tags:
            word = wordtuple[0].lower()
            tag = wordtuple[1]
            if word in glove_dict:
                content_file.write(str(id_curr) + " " + " ".join([str(x) for x in glove_dict[word]]) + " " + str(tag) + "\n") # theoretisch alles mit str.join(statement) möglich
                ids_file.write(str(id_curr) + " " + str(word) + "\n")
                if id_prev != 0:
                    cites_file.write(str(id_prev) + " " + str(id_curr) + "\n")
                id_prev = id_curr
                id_curr += 1
                
    elif small_dataset == True: 
        counter = 0  # also for analysis of brown.content file, otherwise too large to open
        id_curr = 0
        id_prev = 0
        for wordtuple in brown_tags:
            if counter == 20000:
                break
            word = wordtuple[0].lower()
            tag = wordtuple[1]
            if word in glove_dict:
                content_file.write(str(id_curr) + " " + " ".join([str(x) for x in glove_dict[word]]) + " " + str(tag) + "\n") 
                ids_file.write(str(id_curr) + " " + str(word) + "\n")
                if id_prev != 0:
                    cites_file.write(str(id_prev) + " " + str(id_curr) + "\n")
                id_prev = id_curr
                id_curr += 1
                counter += 1


    # code without id's
    #===========================================================================
    # brown_tags = bro.tagged_words()
    #
    # prev_word = ""
    # for wordtuple in brown_tags:
    #     word = wordtuple[0].lower()
    #     tag = wordtuple[1]
    #     if word in glove_dict:
    #         content_file.write(str(word) + " " + " ".join([str(x) for x in glove_dict[word]]) + " " + str(tag) + "\n")
    #         if prev_word != "":
    #             cites_file.write(str(prev_word) + " " + str(word) + "\n")
    #         prev_word = word
    #===========================================================================
    
    
    # old version with uncertain tags
    #===========================================================================
    # brown_corp = bro.sents()
    # brown_tags = dict(bro.tagged_words())
    # 
    # prev_word = ""
    # for line in brown_corp: 
    #     #prev_word = ""     # rein theoretisch müsste das hier falsch sein, weil bei jedem loop zurückgesetzt. funktioniert aber??
    #     for word in line: 
    #         if word.lower() in glove_dict:
    #             #if mmap_content.find(word.lower()) == -1:
    #             content_file.write(str(word.lower()) + " " + " ".join([str(x) for x in glove_dict[word.lower()]]) +" " + str(brown_tags[word]) + "\n")
    #             #mmap_content = mmap.mmap(content_file.fileno(), 0, access=mmap.ACCESS_READ)
    #             if prev_word != "":
    #                 #if mmap_cites.find(prev_word.lower() + " " + word.(lower)) == -1:
    #                 cites_file.write(str(prev_word.lower()) + " " + str(word.lower()) + "\n")
    #                 #mmap_cites = mmap.mmap(cites_file.fileno(), 0, access=mmap.ACCESS_READ)
    #         prev_word = word
    #===========================================================================
    
    
    content_file.flush()
    content_file.close()
    cites_file.flush()
    cites_file.close()
    
    print('Building {} content done.'.format(dataset))
    

# deprecated
#===============================================================================
# def load_data_alt(path="data/brown/", dataset="brown"):
#     print('Loading {} dataset...'.format(dataset))
#     
#     # indexing operator [zeile_anfang : zeile_ende , spalte_anfang : spalte_ende]
#     # glove feature extrahieren
#     idx_features_labels = np.genfromtxt("{}{}.50d.txt".format(path, dataset), )
#     #np.genfromtxt([txt.encode()],delimiter=',',dtype='U20', )
#     features = sp.csr_matrix(idx_features_labels[:, 1:], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, 0])
# 
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.str)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     
#     edges_unordered = np.genfromtxt(brown.sents(), dtype=np.str)
#     #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
# 
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# 
#     print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
# 
#     return features.todense(), adj, labels
#===============================================================================


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32) # rausfinden ob -2 oder -1
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y): #split range ändern training, validation, test
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape