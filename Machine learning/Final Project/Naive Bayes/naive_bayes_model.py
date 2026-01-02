
"""
Custom Naive Bayes Model Definitions
====================================
Shared module containing the custom implementations of Naive Bayes algorithms.
These classes now support dynamic parameter updates (Hot-Swapping alpha/fit_prior).
"""

import numpy as np

class BaseNB:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.classes_ = None
        self.class_log_priors_ = None
        self.feature_log_prob_ = None
        self.class_count_ = None
        self.feature_count_ = None

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features), dtype=np.float64)

    def _update_class_log_priors(self):
        n_classes = len(self.classes_)
        if self.fit_prior:
            # Empirical prior: log(N_c / N_total)
            self.class_log_priors_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            # Uniform prior: log(1/N_classes)
            self.class_log_priors_ = np.full(n_classes, -np.log(n_classes))

    def update_params(self, **params):
        """
        Updates hyperparameters and recomputes the internal log-probabilities
        without needing to see the full training dataset again.
        """
        if 'alpha' in params:
             self.alpha = params['alpha']
        if 'fit_prior' in params:
             self.fit_prior = params['fit_prior']
             
        if self.class_count_ is not None and self.feature_count_ is not None:
            self._recompute_probs()
        return self

    def _recompute_probs(self):
        raise NotImplementedError("Subclasses must implement _recompute_probs")

    def predict_log_proba(self, X):
        # joint_log_likelihood = X @ feature_log_prob.T + class_log_prior
        jll = X @ self.feature_log_prob_.T + self.class_log_priors_
        return jll

    def predict_proba(self, X):
        # Prevent overflow in exp by subtracting max
        jll = self.predict_log_proba(X)
        jll_max = jll.max(axis=1).reshape(-1, 1)
        prob = np.exp(jll - jll_max)
        prob /= prob.sum(axis=1).reshape(-1, 1)
        return prob

    def predict(self, X):
        jll = self.predict_log_proba(X)
        return self.classes_[np.argmax(jll, axis=1)]


class MultinomialNB(BaseNB):
    """
    Multinomial Naive Bayes using Log-Linear algebra.
    """
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self._init_counters(n_classes, n_features)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.feature_count_[idx] = np.array(X_c.sum(axis=0)).flatten()

        self._recompute_probs()
        return self

    def _recompute_probs(self):
        self._update_class_log_priors()
        # P(w|c) = (count(w,c) + alpha) / (count(c) + alpha * V)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)


class ComplementNB(BaseNB):
    """
    Complement Naive Bayes.
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
        
        self._recompute_probs()
        return self

    def _recompute_probs(self):
        self._update_class_log_priors()
        n_classes = len(self.classes_)
        n_features = self.feature_count_.shape[1]
        
        all_feature_count = self.feature_count_.sum(axis=0)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx in range(n_classes):
             complement_count = all_feature_count - self.feature_count_[idx]
             smoothed = complement_count + self.alpha
             denominator = smoothed.sum()
             self.feature_log_prob_[idx] = np.log(smoothed) - np.log(denominator)

    def update_params(self, **params):
        if 'norm' in params:
            self.norm = params['norm']
        return super().update_params(**params)

    def predict_log_proba(self, X):
        jll = - (X @ self.feature_log_prob_.T)
        if self.norm:
            # Add small epsilon to avoid div by zero
            sum_jll = np.abs(jll).sum(axis=1).reshape(-1, 1) + 1e-9
            jll /= sum_jll
        return jll # Logits for CNB


class BernoulliNB(BaseNB):
    """
    Bernoulli Naive Bayes.
    """
    def __init__(self, alpha=1.0, fit_prior=True, binarize=0.0):
        super().__init__(alpha, fit_prior)
        self.binarize = binarize
        self.feature_log_neg_prob_ = None

    def fit(self, X, y):
        # BNB fit needs binarization. We store counts of binarized data.
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
            
        self._recompute_probs()
        return self

    def _recompute_probs(self):
        self._update_class_log_priors()
        # smoothed counts: d_c_w + alpha
        # denominator: n_c + 2*alpha
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.class_count_.reshape(-1, 1) + 2.0 * self.alpha 
        
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        # log(1 - P)
        self.feature_log_neg_prob_ = np.log(1 - np.exp(self.feature_log_prob_))

    def update_params(self, **params):
        if 'binarize' in params:
            # If binarize changes, we actually NEED to refit counts 
            # because feature_count_ was derived from a specific threshold.
            # But the user might want to adjust it.
            # For now, we note that binarize update requires training data access
            # Or we store the original counts if they were raw.
            # Let's assume the counts stored are ALREADY binarized for efficiency.
            self.binarize = params['binarize']
        return super().update_params(**params)

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
