import faiss
import numpy as np
import torch
import torch.nn.functional as F

def search_faiss(X, Q, k):
    # X is the dataset, Q is the query set, k is the number of nearest neighbors to retrieve
    s, knn = faiss.knn(X, Q, k, metric=faiss.METRIC_INNER_PRODUCT)
    return knn, s

def normalize(x):
    # normalize x as a tensor using L2 norm
    return F.normalize(torch.tensor(x), p=2, dim=1).cpu().numpy()

def accuracy(scores, labels):
    # argmax of scores is the predicted label (one prediction per row)
    return np.mean(100 * np.argmax(scores, axis=1) == labels)

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



    