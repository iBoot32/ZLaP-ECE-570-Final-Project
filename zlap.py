import numpy as np
import torch
import faiss
import clip 
from torchvision import datasets
import matplotlib.pyplot as plt
from utils import *
from argparse import ArgumentParser

model, preprocess = clip.load("RN50", device="cpu")

# Combine image and text classifiers knn and similarity scores
#  1. Shift the image knn by the number of classes (avoid overlap)
#  2. Concatenate image and text knn and sim
#  3. Return the concatenated knn and sim
def combine_separate_knns(knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes):
    knn_im = knn_im2im + num_classes
    
    # concat along the columns
    knn = np.concatenate((knn_im, knn_im2text), axis=1)
    sim = np.concatenate((sim_im2im, sim_im2text), axis=1)
    return knn, sim

def encode_image(image):
    with torch.no_grad():
        image_features = model.encode_image(image) # (1, 512) tensor
        image_features /= image_features.norm(dim=-1, keepdim=True) # Normalize features
    return image_features

def get_query_image(query):
    query_image = dataloader[query][0]
    query_image = query_image.unsqueeze(0) # Add batch dimension
    query_feature = model.encode_image(query_image)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)
    query_feature = query_feature.detach().numpy()
    return query_feature

def construct_label_graph(features, clf, k=5):
    # Start by searching for the nearest neighbors of the features (im2im)
    knn_im2im, sim_im2im = search_faiss(features, features, k=k)

    print(f'knn_im2im: {knn_im2im.shape} and {knn_im2im}')
    print(f'sim_im2im: {sim_im2im.shape} and {sim_im2im}')



# performs image_to_image and image_to_text search
# features: each row is a feature vector for an image
# clf: each row is a class vector for an image
def create_separate_graph(features, clf, k):
    num_classes = clf.shape[0]

    # We now use FAISS to search for the nearest neighbors of the features 
    # if k is greater than the number of features or classes, we set k to them
    knn_im2im, sim_im2im = search_faiss(features, features, k=min(k, features.shape[0]))
    knn_im2text, sim_im2text = search_faiss(features, clf, k=min(k, num_classes))

    knn, sim = combine_separate_knns(knn_im2im, sim_im2im, knn_im2text, sim_im2text, num_classes)

    # We now init knn for knn to -1 as a placeholder since we not valid, and sim to 0
    knn_text = np.full((num_classes, knn.shape[1]), -1)
    sim_text = np.zeros((num_classes, sim.shape[1]))
    knn = np.concatenate((knn, knn_text), axis=0)
    sim = np.concatenate((sim, sim_text), axis=0)

    return knn, sim

def do_transductive_lp(features, clf, k, gamma, alpha, scale_sim=False):
    # Create knn and similarity graph for the image and text data
    num_classes = clf.shape[0]
    knn, sim = create_separate_graph(features, clf, k)

    # We then filter for entries in the knn which point to a class, and scale by gamma
    sim[knn < num_classes] = sim[knn < num_classes] ** gamma

    # Turn KNN into laplacian matrix for label propagation
    laplacian = knn_to_laplacian(knn, sim, num_classes)

    # Perform label propagation by solving the linear system via conjugate gradient method
    scores = np.zeros((features.shape[0], num_classes)) # shape (N, num_classes)
    for i in range(num_classes):
        # one-hot encoding where the ith class is 1, then solve for the ith class
        Y = np.zeros((laplacian.shape[0],))
        Y[i] = 1 
        x = conj_gradsearch(laplacian, Y)

        # store results in correct column for class i
        scores[:, i] = x[num_classes:]

# find nearest neighbors between test and unlabeled features and class labels
def get_neighbors_for_inductive(unlabeled_features, clf, test_features, k, gamma, scale_sim=False, xmin=None, xmax=None):
    num_classes = clf.shape[0]

    # searches using faiss (im2im and im2text) between unlabeled and test features, then unlabeld and clf
    knn_im2im, sim_im2im     = search_faiss(unlabeled_features, test_features, k=min(k, unlabeled_features.shape[0]))
    knn_im2text, sim_im2text = search_faiss(unlabeled_features, clf, k=min(k, num_classes))

    # remove entries < 0 
    sim_im2im = np.maximum(sim_im2im, 0)
    sim_im2text = np.maximum(sim_im2text, 0)

    # index shift for im2im and gamma scaling for im2text
    knn_im2im += num_classes
    sim_im2text = sim_im2text ** gamma

    # combine the knn and sim
    test_knn = np.concatenate((knn_im2im, knn_im2text), axis=1)
    test_sim = np.concatenate((sim_im2im, sim_im2text), axis=1)
    return test_knn, test_sim

def do_inductive_lp(unlabeled_features, clf, test_features, k, gamma, alpha):
    num_classes = clf.shape[0]

    # im2im and im2text search for unlabeled and class labels
    knn, sim = create_separate_graph(unlabeled_features, clf, k)

    # gamma scaling for sim: all sim entries with knn < num_classes are scaled by gamma
    sim[knn < num_classes] = sim[knn < num_classes] ** gamma

    # Get Laplacian and find nearest neighbors between test and unlabeled features and class labels
    laplacian = knn_to_laplacian(knn, sim, alpha)
    test_knn, test_sim = get_neighbors_for_inductive(unlabeled_features, clf, test_features, k, gamma)

    # scores for unlabeled features
    scores = np.zeros((unlabeled_features.shape[0], num_classes))
    for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
        # one-hot encoding where the ith class is "s"
        Y = np.zeros((laplacian.shape[0],))
        Y[k] = s
        x = conj_gradsearch(laplacian, Y)
        scores[idx, :] = x[:num_classes]

    return scores.get()

def get_Linv(features, clf, k, gamma, alpha):
    num_classes = clf.shape[0]

    # im2im and im2text search for features and class labels
    knn, sim = create_separate_graph(features, clf, k)

    # gamma scaling for sim: all sim entries with knn < num_classes are scaled by gamma
    sim[knn < num_classes] = sim[knn < num_classes] ** gamma

    # Get Laplacian
    laplacian = knn_to_laplacian(knn, sim, alpha)

    scores = np.zeros((num_classes + features.shape[0], num_classes))
    for i in range(num_classes):
        # one-hot encoding where the ith class is 1, then solve for the ith class
        Y = np.zeros((laplacian.shape[0],))
        Y[i] = 1
        x = conj_gradsearch(laplacian, Y)
        scores[:, i] = x

    return scores.get()

def do_sparse_inductive_lp(unlabeled_features, clf, test_features, k, gamma, alpha):
    num_classes = clf.shape[0]

    # Get Linv
    L_inv = get_Linv(unlabeled_features, clf, k, gamma, alpha)

    # For induction, we need to find the nearest neighbors between test and unlabeled features and class labels
    test_knn, test_sim = get_neighbors_for_inductive(unlabeled_features, clf, test_features, k, gamma)

    Linv_sparse = np.zeros_like(L_inv)
    top = np.argmax(L_inv, axis=1, keepdims=True)
    np.put_along_axis(Linv_sparse, top, np.take_along_axis(L_inv, top, axis=1), axis=1)
    L_inv_sparse = csr_matrix(Linv_sparse)

    # scores for unlabeled features
    # below block of code is given from original implementation due to complexity
    scores = np.zeros((unlabeled_features.shape[0], num_classes))
    for idx, (k, s) in enumerate(zip(test_knn, test_sim)):
        Z = (Linv_sparse[k, :]).copy()
        Z.data = Z.data * s.repeat(np.diff(Z.indptr).get().tolist())
        scores[idx, :] = Z.sum(axis=0)

    return scores.get()



if __name__ == '__main__':
    parser = ArgumentParser(description='ZLAP')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--mode', type=str, default='transductive', choices=['transductive', 'inductive'], help='Transductive or Inductive mode')
    args = parser.parse_args()

    k = args.k
    mode = args.mode

    print(f'  [*] Running ZLaP in {mode} mode with k={k}')



    


    # Load data
    try:
        data = get_data()
    except OSError:
        print("Error loading data")


       

