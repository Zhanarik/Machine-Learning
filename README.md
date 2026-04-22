# Machine Learning Explorations

Welcome to **Machine Learning Explorations**, a collection of **educational Jupyter Notebooks** covering fundamental and advanced concepts in machine learning. This repository is designed for **students, aspiring data scientists, data analysts, and practitioners** who want to learn by doing, experiment with real datasets, and understand the practical challenges behind ML models.

---

## 📚 What You’ll Find

- **Step-by-step explanations** of ML concepts and algorithms  
- **Hands-on code examples** in Python using real-world datasets  
- **Visualizations** to illustrate model behavior, residuals, and performance  
- **Diagnostics and evaluation techniques** for robust and interpretable models  
- **Tips on common pitfalls and remedies** (e.g., multicollinearity, overfitting)  

---

## 🚀 Topics Included

1. **Linear Regression: Understanding, Diagnostics, and Common Pitfalls**  
   - Mathematical foundations and predictions  
   - Residual analysis and assumption checks  
   - Handling multicollinearity (VIF, feature engineering)  
   - Model evaluation: R², Adjusted R², MAE, MSE, AIC, BIC  
   - Comparison of original vs improved model example
  
2. **Logistic Regression: From Log-Odds to Optimization and Regularization**
   - Mathematical foundations (GLM, odds, logit, sigmoid)
   - Log loss (cross-entropy) and convexity proof
   - Gradient descent intuition and LBFGS optimization
   - Effect of feature scaling on convergence
   - L1 vs L2 regularization and coefficient paths
   - Decision boundary geometry and probability interpretation
   - Model evaluation: Log Loss, ROC-AUC, Calibration, Precision/Recall
   - Accuracy paradox and imbalanced data considerations
  
3. **Naive Bayes Classifier**
   - Gaussian Naive Bayes theory and derivation  
   - Step-by-step log-odds calculation example, showing how posterior probabilities are computed  
   - Comparison with Logistic Regression to highlight generative vs discriminative modeling  
   - Small dataset example with calculation of priors, likelihoods, and decision boundary  
   - Code examples with `sklearn` including hyperparameter tuning (alpha)  
   - Visualization of decision boundaries and probability outputs for better intuition  
   - Discussion of assumptions (feature independence, Gaussian distribution) and when NB works well
  
4. **Support Vector Machines — From Intuition to Kernels**
   - Linear classification and decision boundaries
   - Maximum margin intuition
   - Hard and soft margin SVMs with C parameter
   - Support vectors and their role in predictions
   - Limitations of linear boundaries on nonlinear data
   - Feature transformations for separability
   - Kernel trick for efficient nonlinear classification
   - Polynomial and RBF kernels
   - Dual optimization with Lagrange multipliers
   - Sparsity and why only support vectors matter
  
5. **Regularizations - concept, formula and intuition**
   - What is Regularization & Why Do We Need It?
   - The Bias-Variance Tradeoff
   - L2 Regularization (Ridge), L1 Regularization (Lasso)
   - L1 vs L2: Geometric Intuition
   - Elastic Net (L1 + L2)
   - Regularization in Neural Networks: Dropout, Batch Normalization, Weight Decay, Early Stopping, Data Augmentation
   - Bayesian Interpretation
   - Interview Questions & Answers
  
6. **Decision Trees — Splitting, Pruning, and Complexity Control**
   - Impurity measures: Gini, Entropy, and MSE with mathematical derivations
   - Step-by-step manual split calculation with Gini gain at every threshold
   - Decision boundary visualization at increasing depths (underfitting → overfitting)
   - Tree structure visualization and node interpretation
   - Pre-pruning hyperparameters (max_depth, min_samples_split, min_samples_leaf)
   - Cost-complexity post-pruning with alpha path and cross-validation
   - Strengths and weaknesses: interpretability vs high variance

7. **Random Forest — Bagging and Feature Randomness**
   - Bias-variance decomposition and why single trees are unstable
   - Bootstrap sampling and the bagging framework
   - Feature randomness: decorrelating trees to reduce variance (with correlation formula)
   - Comparison of Single Tree vs Bagging vs Random Forest decision boundaries
   - Out-of-Bag (OOB) error as free validation (with the 1/e ≈ 37% derivation)
   - Number of trees vs performance (diminishing returns, no overfitting)
   - Feature importance: MDI (Gini-based) vs Permutation Importance with bias discussion

8. **Boosting — AdaBoost, Gradient Boosting, and XGBoost**
   - AdaBoost algorithm: sample re-weighting, learner weights, and weighted vote
   - AdaBoost from scratch with sample weight evolution visualization
   - Gradient Boosting algorithm: fitting residuals as negative gradients
   - Gradient Boosting from scratch (regression) with residual shrinkage visualization
   - Gradient Boosting for classification: log-odds, sigmoid, and pseudo-residuals
   - Learning rate (shrinkage) tradeoff: small η + many trees = best generalization
   - XGBoost enhancements: regularized objective, second-order approximation, column subsampling
   - Side-by-side decision boundary and learning curve comparisons across all methods
   - When to use which: practical decision guide for interviews
   - Key hyperparameters cheat sheet for Decision Trees, Random Forest, and Gradient Boosting

9. **Dimensionality Reduction — PCA, t-SNE, UMAP and Beyond**
   * Why reduce dimensions: curse of dimensionality, visualization, noise removal
   * PCA: covariance matrix, eigendecomposition, explained variance ratio
   * Step-by-step PCA from scratch with NumPy
   * Scree plot and choosing the number of components
   * t-SNE: perplexity, KL divergence, and crowding problem
   * UMAP: graph-based approach and comparison with t-SNE
   * Side-by-side visualization comparisons on real datasets
   * When to use which: practical decision guide

10. **Clustering — K-Means, Hierarchical, DBSCAN, GMM, and Spectral**
    * What is clustering: types (partitional, hierarchical, density-based, model-based)
    * K-Means: intuitive explanation, math (WCSS objective, convergence proof), K-Means++ initialization
    * K-Means from scratch implementation with NumPy
    * Choosing K: Elbow method, Silhouette analysis, Calinski-Harabasz, Davies-Bouldin
    * Hierarchical clustering: linkage criteria (single, complete, average, Ward) with dendrograms
    * DBSCAN: core/border/noise points, eps selection via k-distance plot
    * Gaussian Mixture Models: EM algorithm, covariance types, BIC/AIC model selection
    * Spectral clustering: graph Laplacian intuition and non-convex shape handling
    * Evaluation metrics: Silhouette, ARI, NMI (internal vs external)
    * Full algorithm comparison across challenging datasets (moons, circles, anisotropic)
    * Interview questions and answers
11. **Neural Networks — From Perceptron to LSTMs**
    * Perceptron: single neuron model, learning algorithm, and the XOR limitation
    * Multi-Layer Perceptron (MLP): Universal Approximation Theorem, solving XOR from scratch
    * Backpropagation: full chain-rule derivation, computational graphs, and gradient checking
    * Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, Swish — comparison and derivatives
    * Vanishing and exploding gradients: causes, simulation, and solutions (skip connections, proper initialization)
    * Weight initialization: Xavier/Glorot vs He/Kaiming with activation distribution visualization
    * Optimizers: SGD, Momentum, Nesterov, Adam — update equations and trajectory comparison on Rosenbrock surface
    * Learning rate scheduling: step decay, cosine annealing, warm-up, one-cycle policy
    * Batch Normalization: algorithm, training vs inference, LayerNorm/GroupNorm/RMSNorm alternatives
    * Training pipeline: complete training loop, loss functions (MSE, cross-entropy, focal, contrastive), hyperparameter tuning
    * Overfitting and regularization: L1/L2, Dropout (from scratch), early stopping, label smoothing, data augmentation
    * Transfer learning: feature extraction, fine-tuning, LoRA and parameter-efficient methods
    * CNNs: convolution from scratch, output size formula, landmark architectures (LeNet → ResNet → ViT)
    * RNNs: vanilla RNN limitations, Backpropagation Through Time (BPTT), gradient flow analysis
    * LSTMs: gate equations (forget, input, output), cell state as gradient highway, GRU comparison
    * Sequence models: Bidirectional RNNs, Seq2Seq, Attention mechanism, and why Transformers won
    * Full spiral dataset classification example trained from scratch with Adam optimizer
    * Interview rapid-fire Q&A and preparation checklist

 

---
## ✨ Note to Users

All notebooks are intended for educational purposes. They include detailed explanations and visualizations so you can learn by experimenting. 
Whether you’re a beginner or an intermediate learner, you’re welcome to use these notebooks as a practical learning resource.
