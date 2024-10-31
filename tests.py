from zlap import *
import zlap_original.zlap as their_zlap
import zlap_original.utils as their_utils

# COPYRIGHT AND CREDITS NOTICE
#   THIS CODE IS PURELY TEST CODE AND IS NOT PART OF THE ORIGINAL IMPLEMENTATION
#   


def check(func):
    print(f"\033[93m" + f"Testing {func.__name__}..." + "\033[0m")
    def wrapper(*args, **kwargs):
        if func(*args, **kwargs):
            print("\033[92m" + "Test passed successfully!" + "\033[0m")
        else:
            print("\033[91m" + f"Test failed" + "\033[0m")
    return wrapper

def test_graph_norm(n=15):
    # Define test cases with different adjacency matrices
    test_cases = [
        np.array([[0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [1, 0, 0, 1],
                   [0, 1, 1, 0]]),  
        
        np.array([[0, 1, 0, 0],
                   [1, 0, 1, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0]]), 

        np.array([[0, 1, 0],
                   [1, 0, 1],
                   [0, 1, 0]]),

        np.array([[0, 1, 0, 0, 0],
                   [1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1],
                   [0, 1, 0, 0, 1],
                   [0, 0, 1, 1, 0]]),

        # Random graph with n nodes
        np.random.randint(0, 2, size=(n, n)).astype(int),
        np.random.randint(0, 20, size=(2*n, 2*n)).astype(int)
    ]

    all_tests_passed = True  # Flag to track overall test status

    for i, graph in enumerate(test_cases):
        print(f"  [*] Testing graph norm case {i + 1}...")

        # Normalize
        normalized_graph_1 = normalize_connection_graph(graph)
        normalized_graph_2 = their_utils.normalize_connection_graph(graph)

        # Check if outputs are close
        if not np.allclose(normalized_graph_1.toarray(), normalized_graph_2.toarray(), atol=1e-6):
            print(f"    - Test failed: Normalized graphs are not close!")
            all_tests_passed = False

    return all_tests_passed  # Return the overall test result


def test_knn_to_laplacian(n=5):
    # Define test cases with different KNN and similarity score arrays
    test_cases = [
        (np.array([[1, 2], [0, 3], [0, 1], [1, 2]]), np.array([0.9, 0.8, 0.7, 0.6])),
        (np.array([[1, 2], [0, 3], [0, -1], [1, 2]]), np.array([0.9, 0.8, 0.0, 0.6])),
        (np.array([[1, 2], [0, -1], [0, 1]]), np.array([0.9, 0.0, 0.8])),
        (np.array([[1, 2, 3], [0, 2, -1], [0, 1, 3], [0, 1, 2]]), np.array([0.9, 0.8, 0.7, 0.6])),
        (np.random.randint(0, n, size=(n, 2)), np.random.rand(n))
    ]

    all_tests_passed = True  # Flag to track overall test status
    for i, (knn, sim) in enumerate(test_cases):
        print(f"\nTesting case {i + 1}:")

        # Normalize knn to ensure valid inputs
        valid_knn_mask = knn != -1  # Mask for valid KNN entries
        valid_knn_count = np.sum(valid_knn_mask, axis=1)  # Count of valid neighbors per row

        # Prepare the similarity array based on valid KNN entries
        valid_sim = np.repeat(sim, valid_knn_count).flatten()[:np.sum(valid_knn_count)]

        # Create a sparse representation of the valid KNN
        knn_flat = knn[valid_knn_mask]  # Keep only valid knn entries
        row_indices = np.repeat(np.arange(len(knn)), knn.shape[1])[valid_knn_mask.flatten()]

        if knn_flat.size > 0:
            knn_csr = csr_matrix((valid_sim, (row_indices, knn_flat)), shape=(len(knn), n))
        else:
            knn_csr = csr_matrix((len(knn), n))

        normalized_graph_2 = their_utils.knn2laplacian(knn, sim)
        normalized_graph_1 = knn_to_laplacian(knn_csr, valid_sim)

        # Compare the outputs
        print("Normalized Graph 1:\n", normalized_graph_1.toarray())
        print("Normalized Graph 2:\n", normalized_graph_2.toarray())

        # Check if outputs are close
        if not np.allclose(normalized_graph_1.toarray(), normalized_graph_2.toarray(), atol=1e-6):
            print(f"    - Test failed: Normalized graphs are not close!")
            all_tests_passed = False

    return all_tests_passed  # Return the overall test result

def test_create_separate_graph():
    # read data
    (
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        test_targets,
        clf_text,
        clf_image_train,
        clf_image_val,
        clf_image_test,
        clf_cupl_text,
        clf_cupl_image_train,
        clf_cupl_image_val,
        clf_cupl_image_test
    ) = get_data()

    # Create separate graphs
    knn1, sim1 = create_separate_graph(train_features, clf_text, 5)
    knn2, sim2 = their_zlap.create_separate_graph(train_features, clf_text, 5)

    # Compare the outputs
    print("KNN 1 shape:", knn1.shape)
    print("KNN 2 shape:", knn2.shape)
    print("Similarity 1 shape:", sim1.shape)
    print("Similarity 2 shape:", sim2.shape)

    # Check if outputs are close
    if not np.allclose(knn1, knn2) or not np.allclose(sim1, sim2) or not knn1.shape == knn2.shape or not sim1.shape == sim2.shape:
        print(f"    - Test failed: KNN and similarity scores are not close!")
        return False
    return True

def test_knn_to_laplacian():
        # read data
    (
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        test_targets,
        clf_text,
        clf_image_train,
        clf_image_val,
        clf_image_test,
        clf_cupl_text,
        clf_cupl_image_train,
        clf_cupl_image_val,
        clf_cupl_image_test
    ) = get_data()

    # Create separate graphs
    knn1, sim1 = create_separate_graph(train_features, clf_text, 5)
    knn2, sim2 = their_zlap.create_separate_graph(train_features, clf_text, 5)

    features = train_features
    clf = clf_text
    gamma = 0.5

    num_classes = clf.shape[0]
    sim1[knn1 < num_classes] = sim1[knn1 < num_classes] ** gamma
    laplacian = knn_to_laplacian(knn1, sim1, num_classes)

    sim2[knn2 < num_classes] = sim2[knn2 < num_classes] ** gamma
    laplacian2 = their_zlap.knn2laplacian(knn2, sim2, num_classes)

    # Compare the outputs
    print("Laplacian 1 shape:", laplacian.shape)
    print("Laplacian 2 shape:", laplacian2.shape)

    # compare csr matrix
    if not csr_matrices_are_similar(laplacian, laplacian2):
        print(f"    - Test failed: Laplacians are not similar!")
        return False
    return True

def csr_matrices_are_similar(csr_matrix1, csr_matrix2, tolerance=1e-9):
    # Step 1: Check if shapes match
    if csr_matrix1.shape != csr_matrix2.shape:
        return False
    if csr_matrix1.data.shape != csr_matrix2.data.shape:
        return False

    # Step 2: Check data, indices, and indptr
    data_similar = np.allclose(csr_matrix1.data, csr_matrix2.data, atol=tolerance)
    
    return data_similar 



if __name__ == '__main__':
    check(test_graph_norm)()
    # check(test_knn_to_laplacian)()
    check(test_create_separate_graph)()
    check(test_knn_to_laplacian)()