import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

# test imports
from scipy.sparse import csr_matrix, diags

def knn_to_laplacian(knn, s, alpha=0.99):
    # Extract dimensions
    N, k = knn.shape

    # Create row indices repeated k times
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T

    # Flatten arrays column-wise
    knn_flat = knn.flatten("F")
    row_idx_rep_flat = row_idx_rep.flatten("F")
    sim_flat = s.flatten("F")

    # Filter out invalid entries
    valid_knn = np.where(knn_flat != -1)[0]
    knn_flat = knn_flat[valid_knn]
    row_idx_rep_flat = row_idx_rep_flat[valid_knn]
    sim_flat = sim_flat[valid_knn]

    # Create sparse adjacency matrix W
    W = csr_matrix((sim_flat, (row_idx_rep_flat, knn_flat)), shape=(N, N))
    W = W + W.T  # Make W symmetric

    # Normalize W to create Wn
    Wn = normalize_connection_graph(W)

    # Construct Laplacian matrix L
    L = np.eye(Wn.shape[0]) - alpha * Wn
    return L


def search_faiss(X, Q, k):
    # X is the dataset, Q is the query set, k is the number of nearest neighbors to retrieve
    s, knn = faiss.knn(X, Q, k, metric=faiss.METRIC_INNER_PRODUCT)
    return knn, s

def normalize(x):
    # normalize x as a tensor using L2 norm
    return F.normalize(torch.tensor(x), p=2, dim=1).cpu().numpy()

def accuracy(scores, labels):
    # argmax of scores is the predicted label (one prediction per row)
    acc = np.mean(100.0 * (np.argmax(scores, axis=1) == labels))
    return acc

def get_data():
    base_path = "zlap_features/features/caltech101"
    model = "RN50"
    
    # Load features and targets
    train_features = np.load(f"{base_path}/{model}_openai_train_feats.npy")
    train_targets = np.load(f"{base_path}/{model}_openai_train_targets.npy")
    val_features = np.load(f"{base_path}/{model}_openai_val_feats.npy")
    val_targets = np.load(f"{base_path}/{model}_openai_val_targets.npy")
    test_features = np.load(f"{base_path}/{model}_openai_test_feats.npy")
    test_targets = np.load(f"{base_path}/{model}_openai_test_targets.npy")
    
    # Load classifiers
    clf_text = np.load(f"{base_path}/classifiers/{model}_openai_text_classifier.npy").T
    clf_cupl_text = np.load(f"{base_path}/classifiers/{model}_openai_cupl_text_classifier.npy").T
    clf_image_train = np.load(f"{base_path}/classifiers/{model}_openai_inmap_proxy_classifier_train.npy").T
    clf_cupl_image_train = np.load(f"{base_path}/classifiers/{model}_openai_cupl_inmap_proxy_classifier_train.npy").T
    clf_image_val = np.load(f"{base_path}/classifiers/{model}_openai_inmap_proxy_classifier_val.npy").T
    clf_cupl_image_val = np.load(f"{base_path}/classifiers/{model}_openai_cupl_inmap_proxy_classifier_val.npy").T
    clf_image_test = np.load(f"{base_path}/classifiers/{model}_openai_inmap_proxy_classifier_test.npy").T
    clf_cupl_image_test = np.load(f"{base_path}/classifiers/{model}_openai_cupl_inmap_proxy_classifier_test.npy").T

    # Return normalized values
    return (
        normalize(train_features.astype(np.float32)),
        train_targets,
        normalize(val_features.astype(np.float32)),
        val_targets,
        normalize(test_features.astype(np.float32)),
        test_targets,
        normalize(clf_text),
        normalize(clf_image_train),
        normalize(clf_image_val),
        normalize(clf_image_test),
        normalize(clf_cupl_text),
        normalize(clf_cupl_image_train),
        normalize(clf_cupl_image_val),
        normalize(clf_cupl_image_test),
    )

# Accuracy and Precision Classifiers
#   voc_ap is average precision as defined in the PASCAL VOC challenge
#     1. modify precision and recal arrays with some padding
#     2. smooth the precision by iterating backwards and taking the maximum value bewteen curr and next (i-1)
#     3. find the indices of the unique recall values
#     4. apply trapzoidal rule to calculate the area under the curve
def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    change_indices = np.array([], dtype=int)
    for i in range(1, mrec.size):
        if mrec[i] != mrec[i - 1]:
            change_indices = np.append(change_indices, i - 1)
    
    ap = np.sum((mrec[change_indices + 1] - mrec[change_indices]) * mpre[change_indices + 1])
    return ap

def voc_mAP(img_cls, num_classes):
    ground_truth = img_cls[:, num_classes].astype(np.float32)

    # initialize variables
    num_samples = len(ground_truth)
    true_pos = np.zeros(num_samples)
    false_pos = np.zeros(num_samples)
    avg_precision_per_class = []

    for class_idx in range(num_classes):
        # extract confidence and sort by confidence
        conf = img_cls[:, class_idx].astype(np.float32)
        sorted_indices = np.argsort(-conf)
        sorted_labels = [ground_truth[i][class_idx] for i in sorted_indices]

        # compute true positive and false positive
        for i in range(num_samples):
            true_pos[i] = sorted_labels[i] > 0
            false_pos[i] = 1 - true_pos[i] <= 0

        # compute recall (true_pos/num_pos) and precision (true_pos/(true_pos+false_pos))
        true_num = np.sum(true_pos)
        rec = np.cumsum(true_pos) / true_num
        prec = np.cumsum(true_pos) / (np.cumsum(true_pos) + np.cumsum(false_pos), np.finfo(np.float64).eps) # avoid division by zero

        # compute average precision
        avg_precision = voc_ap(rec, prec, true_num)
        avg_precision_per_class.append([avg_precision])
        avg_precision_per_class = np.array(avg_precision_per_class) * 100
        mAP = np.mean(avg_precision_per_class)
        
        return mAP

# Normalize Connection Graph
#  We use symmetric normalization to normalize the connection (adjacency) graph as defined:
#     norm(G) = D^(-1/2) * G * D^(-1/2)
#  where D = diag(G1 + G2 + ... + Gn) is the degree matrix for each degree of the graph
#  and G (graph) is the adjacency matrix of the graph
def normalize_connection_graph(graph):
    # Authors of ZLaP convert matrices to Compressed Sparse Row format for faster computation
    graph = csr_matrix(graph)

    # Because we only care about non-self connections, we may remove all diagonal entries
    graph = graph - sparse.diags(graph.diagonal())

    # We then construct the degree vector. `graph` is an adjacency matrix so we can sum along the columns
    degree_vec = np.array(graph.sum(axis=1))
    assert degree_vec.shape == (graph.shape[0], 1)
    degree_vec[degree_vec == 0] = 1 # Avoid division by zero
    assert(not any(degree_vec < 0)) # Degree vector should not contain negative values

    # We then compute the inverse of the square root of the degree vector and add each entry to the diagonal matrix
    # This is faster than computing the inverse of the degree matrix
    degree_vec_inv_sqrt = np.power(degree_vec, -0.5)

    # Make inv_sqrt vector into diagonal matrix
    degree_inv_sqrt_diag = sparse.diags(degree_vec_inv_sqrt.flatten())

    # Conduct symmetric normalization
    graph_normalized = degree_inv_sqrt_diag @ graph @ degree_inv_sqrt_diag
    return graph_normalized

# Used in transductive inference, we solve linear system Lx=Y via conjugate gradient method
# Reasoning given in paper as it's faster than other iterative methods
# L is the Laplacian matrix and Y is the target matrix
def conj_gradsearch(L, Y, tol=1e-6, maxiter=50):
    x, _ = sparse.linalg.cg(L, Y, tol=tol, maxiter=maxiter)
    return x