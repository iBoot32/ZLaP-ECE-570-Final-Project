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


       

