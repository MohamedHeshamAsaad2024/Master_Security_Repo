
"""
Custom Naive Bayes Model Definitions
====================================
Shared module containing the custom implementations of Naive Bayes algorithms.
Separating this allows consistent pickling/unpickling between training and GUI.
"""

import numpy as np

class BaseNB:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.classes_ = None
        self.class_log_priors_ = None
        self.feature_log_prob_ = None

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

    def _update_class_log_priors(self, n_classes):
        if self.fit_prior:
            # Empirical prior: log(N_c / N_total)
            self.class_log_priors_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            # Uniform prior: log(1/N_classes)
            self.class_log_priors_ = np.full(n_classes, -np.log(n_classes))

    def predict_log_proba(self, X):
        # joint_log_likelihood = X @ feature_log_prob.T + class_log_prior
        jll = X @ self.feature_log_prob_.T + self.class_log_priors_
        return jll

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        jll = self.predict_log_proba(X)
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB(BaseNB):
    """
    Multinomial Naive Bayes using Log-Linear algebra.
    Good for word counts/TF-IDF.
    """
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self._init_counters(n_classes, n_features)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            # Sum tf-idf weights or counts
            self.feature_count_[idx] = np.array(X_c.sum(axis=0)).flatten()

        self._update_class_log_priors(n_classes)

        # Compute Feature Log Probabilities with Smoothing
        # P(w|c) = (count(w,c) + alpha) / (count(c) + alpha * V)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        
        return self


class ComplementNB(BaseNB):
    """
    Complement Naive Bayes.
    Describes the "complement" class (all classes EXCEPT c).
    Often performs better on imbalanced datasets.
    """
    def __init__(self, alpha=1.0, fit_prior=True, norm=False):
        super().__init__(alpha, fit_prior)
        self.norm = norm

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self._init_counters(n_classes, n_features)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.feature_count_[idx] = np.array(X_c.sum(axis=0)).flatten()
        
        self._update_class_log_priors(n_classes) 

        all_feature_count = self.feature_count_.sum(axis=0)
        
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx in range(n_classes):
             complement_count = all_feature_count - self.feature_count_[idx]
             smoothed = complement_count + self.alpha
             denominator = smoothed.sum()
             self.feature_log_prob_[idx] = np.log(smoothed) - np.log(denominator)

        return self

    def predict_log_proba(self, X):
        jll = - (X @ self.feature_log_prob_.T)
        if self.norm:
            sum_jll = jll.sum(axis=1).reshape(-1, 1)
            jll /= np.abs(sum_jll)
        return jll


class BernoulliNB(BaseNB):
    """
    Bernoulli Naive Bayes.
    Binarizes input and uses multivariate Bernoulli distribution.
    """
    def __init__(self, alpha=1.0, fit_prior=True, binarize=0.0):
        super().__init__(alpha, fit_prior)
        self.binarize = binarize
        self.feature_log_neg_prob_ = None

    def fit(self, X, y):
        if self.binarize is not None:
            X_binary = (X > self.binarize).astype(int)
        else:
            X_binary = X
            
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self._init_counters(n_classes, n_features)
        
        for idx, c in enumerate(self.classes_):
            X_c = X_binary[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.feature_count_[idx] = np.array(X_c.sum(axis=0)).flatten()
            
        self._update_class_log_priors(n_classes)

        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.class_count_.reshape(-1, 1) + 2.0 * self.alpha 
        
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.feature_log_neg_prob_ = np.log(1 - np.exp(self.feature_log_prob_))
        
        return self

    def predict_log_proba(self, X):
        if self.binarize is not None:
             X_bin = (X > self.binarize).astype(np.float64)
        else:
             X_bin = X
        
        jll = []
        for idx in range(len(self.classes_)):
            w = self.feature_log_prob_[idx] - self.feature_log_neg_prob_[idx]
            const = self.feature_log_neg_prob_[idx].sum() + self.class_log_priors_[idx]
            score = X_bin @ w + const
            jll.append(score)
            
        return np.array(jll).T
