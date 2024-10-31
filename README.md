# ZLaP-ECE-570-Final-Project

This is a reimplementation and extension of **Vladan Stojnić, Yannis Kalantidis, Giorgos Tolias, "Label Propagation for Zero-shot Classification with Vision-Language Models", In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.**

Code is [publically-available online](https://github.com/vladan-stojnic/ZLaP/tree/main) and was used to gather a list of functions and as a reference for math-intensive functions not described programmatically in the paper such as graph creation. As such my code will bear strong resemblance for the code functions, and I will compare and evaluate my code to theirs in my course paper. I have also planned an extension: A practical pipeline using ZLaP for retail applications. 

As described in my project proposal, imagine this scenario:

> You work in retail distribution, so the classes/objects will change often enough to make manual data gathering and labeling infeasible, and these objects are not detectable with classical computer vision. Zero-shot classification would allow you to classify these objects based simply on a class label and some unlabeled images of your objects. This is especially valuable when dealing with new product lines or seasonal items that may not have been seen when the model was first deployed.
>
> Furthermore, ZLaP does not require a training step so it can be deployed without extensive prerequisite setup time, especially since the label propagation is based on KNN which does not require deployment on a large GPU cluster.

I created a plan/order for function implementation based on their simplicity and usage in other functions:

| **Category**       | **Functionality**                | **Status** | **Completion Date**     |
|--------------------|----------------------------------|------------|-------------------------|
| **Utils**          | normalize                       | ✔️         | Oct 29, 2024            |
|                    | accuracy                        | ✔️         | Oct 29, 2024            |
|                    | get_data                        | ✔️         | Oct 29, 2024            |
|                    | voc_ap                          | ✔️         | Oct 29, 2024            |
|                    | voc_mAP                         | ✔️         | Oct 29, 2024            |
|                    | search_faiss                    | ✔️         | Oct 29, 2024            |
|                    | normalize connection graph      | ✔️         | Oct 30, 2024        |
|                    | laplacian                       | ✔️         | Oct 30, 2024         |
|                    | dfs                             | ✔️         | Oct 30, 2024         |
| **Main Functions** | get_args                        | ✔️         | Oct 29, 2024            |
|                    | combine_separate_knns           | ✔️         | Oct 29, 2024            |
|                    | create_separate_graph           | ⬜         |                         |
|                    | do_transductive_lp              | ⬜         |                         |
|                    | get_neighbors_for_inductive     | ⬜         |                         |
|                    | do_inductive_lp                 | ⬜         |                         |
|                    | get_Linv                        | ⬜         |                         |
|                    | do_sparse_inductive_lp          | ⬜         |                         |
|                    | main                            | ⬜         |                         |
| **Expansions**     | Train on custom data            | ⬜         |                         |
|                    | Evaluate on retail data         | ⬜         |                         |
