import numpy as np
import torch
import faiss
import clip 
from torchvision import datasets
import matplotlib.pyplot as plt
from utils import *

model, preprocess = clip.load("RN50", device="cpu")

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



if __name__ == '__main__':

    print("starting...")

    # Load data
    try:
        data = get_data()
    except OSError:
        print("Error loading data")


    # features, classes = load_data()
    # dataloader = datasets.Caltech101(root='data', download=True, transform=preprocess)

    # # FAISS is built on "indexes" which are used to store and query the data
    # # Create an index using FAISS, using shape[1] (columns) as the dimensionality 
    # index = faiss.IndexFlatL2(features.shape[1])
    # index.add(features)

    # construct_label_graph(features, classes, 5)

    # # random number 0-10000
    # query = np.random.randint(0, 5000)
    # query_feature = get_query_image(query)

    # # Search for the nearest neighbors of the query image
    # D, I = index.search(query_feature, 5)

    # # print distances and indexes
    # print(D)
    # print(I)

    # # show original image
    # img, _class = dataloader[query]
    # idx_to_class = {i: category for i, category in enumerate(dataloader.categories)}
    # img = img.permute(1, 2, 0).numpy()
    # plt.imshow(img)
    # plt.title(f"ORIGINAL class: {idx_to_class[_class]} with image idx {query}")
    # plt.show()


    # for z in range(5):
    #     img, _class = dataloader[classes[I[0][z]]]
    #     idx_to_class = {i: category for i, category in enumerate(dataloader.categories)}

    #     img = img.permute(1, 2, 0).numpy()
    #     plt.imshow(img)
    #     plt.title(f"Predicted class: {idx_to_class[_class]} with image idx {_class}")
    #     plt.show()



       

