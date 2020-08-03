import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from baycomp import two_on_single

# Resources:
# https://christophm.github.io/interpretable-ml-book/

class Compare:
    def __init__(self):
        pass

    def repeated_kfold(pipe1, pipe2, X, y, n_repeats=3, k=5, random_state=0, stratify=True):
        """
        Performed repeated k-fold cross validation to get scores.

        Parameters
        ----------
        pipe1 : sklearn.pipeline.Pipeline or BaseEstimator
            Any pipeline or estimator that implements the fit() and score() methods.
        pipe2 : sklearn.pipeline.Pipeline or BaseEstimator
            Pipeline to compare against.
        X : ndarray
            Array of features.
        y : ndarray
            Array of outputs to predict.
        n_repeats : int
            Number of times cross-validator needs to be repeated.
        k : int
            K-fold cross-validation to use.
        random_state : int or RandomState instance
            Controls the randomness of each repeated cross-validation instance.
        stratify : bool
            If True, use RepeatedStratifiedKFold - this is only valid for classification tasks.

        Returns
        -------
        scores1, scores2
            List of scores for each pipeline.
        """
        from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
        if (stratify):
            rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=random_state)
        else:
            rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=random_state)
        
        X = np.array(X)
        y = np.array(y)

        scores1 = []
        scores2 = []
        for train_index, test_index in rkf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            pipe1.fit(X_train, y_train)
            scores1.append(pipe1.score(X_test, y_test))

            pipe2.fit(X_train, y_train)
            scores2.append(pipe2.score(X_test, y_test))

        return scores1, scores2

    def corrected_t(scores1, scores2, n_repeats):
        """
        Performs 1-sided hypothesis testing to see if any difference in pipelines' peformances
        are statisically significant using a correlated, paired t-test with the Nadeau & Bengio (2003)
        correction. The test checks if the first pipeline is superior to the second using the alternative
        hypothesis, H1: mean(scores1-scores2) > 0.

        Notes
        -----
        Reject H0 (that pipelines are equally good) in favor of H1 (pipeline1 is better) if p < alpha, otherwise fail to reject H0 (not enough evidence to suggest they are different).
        The formulation of this test is that pipeline1 has the best (average) performance or score of the two, and you want to check if that is statistically significant or not.

        Parameters
        ----------
        scores1 : array-like
            List of scores from pipeline 1.
        scores2 : array-like
            List of scores from pipeline 2.
        n_repeats : int
            Number of times cross-validator was repeated.

        Returns
        -------
        p
            P value
        """
        assert(len(scores1) == len(scores2)), 'scores must have the same overall length'
        k_fold = len(scores1) // int(n_repeats)
        n = k_fold*n_repeats
        assert(n == len(scores1)), 'scores must be divisible by n_repeats'

        rho = 1.0/k_fold
        performance_diffs = np.array(scores1) - np.array(scores2) # H1: mu > 0
        corrected_t = (np.mean(performance_diffs) - 0.0) / np.sqrt((1.0/n + rho/(1.0-rho))*(np.std(performance_diffs, ddof=1)**2))
        
        return 1.0 - scipy.stats.t.cdf(x=corrected_t, df=n-1) # 1-sided test


    def bayesian_comparison(scores1, scores2, n_repeats, rope=0):
        """
        Performs Bayesian analysis to predict the probability that pipe(line)1 outperforms
        pipe(line)2 based on repeated kfold cross validation results using a correlated t-test.

        If prob[X] > 1.0 - alpha, then you make the decision that X is better.
        If no prob's reach this threshold, make no decision about the super(infer)iority
        of the pipelines relative to each other.

        Notes
        -----
        See https://baycomp.readthedocs.io/en/latest/functions.html.

        Parameters
        ----------
        scores1 : array-like
            List of scores from each repeat of each CV fold for pipe1.
        scores2 : array-like
            List of scores from each repeat of each CV fold for pipe2.
        n_repeats : int
            Number of repetitions of cross validation.
        rope : float
            The width of the region of practical equivalence.

        Returns
        -------
        probs
            Tuple of (prob_1, p_equiv, prob_2)
        """
        scores1 = np.array(scores1).flatten()
        scores2 = np.array(scores2).flatten()
        probs = two_on_single(scores1, scores2, rope=rope, runs=n_repeats, names=None, plot=False)

        if (rope == 0):
            probs = np.array([probs[0], 0, probs[1]])
        
        return probs > (1.0-alpha), probs

class InspectModel:
    def __init__(self):
        pass
    
    @staticmethod
    def confusion_matrix(model, X, y_true):
        """
        For comparing classification models based on true/false positive rates.

        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.
        """
        from sklearn.metrics import confusion_matrix

        confmat = confusion_matrix(y_true=y_true, y_pred=model.predict(X))

        fig = plt.figure()
        _ = sns.heatmap(confmat, ax=plt.gca(), annot=True, xticklabels=model.classes_, 
                        yticklabels=model.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        return plt.gca()
    
    @staticmethod
    def roc_curve(model, X, y, n_splits=10):
        """
        For selecting classification models based on true/false positive rates.

        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.
        """
        from sklearn.metrics import roc_curve, auc

        fig = plt.figure()
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        cv = list(StratifiedKFold(n_splits=n_splits, random_state=0).split(X, y))

        for i, (train, test) in enumerate(cv):
            probas = model.fit(X[train], y[train]).predict_proba(X[test])

            fpr, tpr, thresholds = roc_curve(y[test],
                                            probas[:, 1],
                                            pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,
                    tpr,
                    label='ROC fold %d (area = %0.2f)'
                          % (i+1, roc_auc))

        plt.plot([0, 1],
                [0, 1],
                linestyle='--',
                color=(0.6, 0.6, 0.6),
                label='Random guessing')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.plot([0, 0, 1],
                [0, 1, 1],
                linestyle=':',
                color='black',
                label='Perfect performance')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend(loc='best')

        plt.tight_layout()

        return plt.gca()

    @staticmethod
    def learning_curve(self, model, X, y, train_sizes=np.linspace(0.1, 1, 10), cv=10):
        """
        For diagnosing bias/variance issues in a model. 
        The validation and training accuracy curves should converge "quickly" 
        (if not, high variance) and to a "high" accuracy (if not, high bias).
        If it doesn't converge, it probably needs more data to train on.

        See Ch. 6 of "Python Machine Learning" by Raschka & Mirjalili.

        https://scikit-learn.org/stable/modules/learning_curve.html

        Example
        -------
        >>> pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1))
        >>> learning_curve(pipe_lr, X_train, y_train)
        """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=model,
                                   X=X,
                                   y=y,
                                   train_sizes=train_sizes,
                                   cv=cv, # Stratified by default in scikit-learn
                                   n_jobs=1)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_mean,
                color='blue', marker='o',
                markersize=5, label='Training accuracy')

        plt.fill_between(train_sizes,
                        train_mean + train_std,
                        train_mean - train_std,
                        alpha=0.15, color='blue')

        plt.plot(train_sizes, test_mean,
                color='green', linestyle='--',
                marker='s', markersize=5,
                label='Validation accuracy')

        plt.fill_between(train_sizes,
                        test_mean + test_std,
                        test_mean - test_std,
                        alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.tight_layout()

        return plt.gca()
    
    @staticmethod
    def plot_residuals(y_true, y_pred):
        """
        Plot residuals and fit to a Gaussian distribution.  A good fit might indicate all 
        predictive "information" has been extracted and the remaining uncertainty
        is due to random noise.

        Parameters
        ----------
        y_true : ndarray
          N x K array of N observations made of K outputs.
        y_pred : ndarray
          N x K array of N predictions of K variables. A model with a scalar output, for example, is just a column vector (K=1).
        """
        n_vars = y_true.shape[1]
        assert(y_true.shape[1] == y_pred.shape[1])

        for i in range(n_vars):
            sns.jointplot(x=y_true[:,i], y=y_pred[:,i], kind='resid')

        return plt.gca()
    
    @staticmethod
    def pdp(model, X, features, **kwargs):
        """
        Partial dependence plots for features in X.
        
        Partial dependence plots (PDP) show the dependence between the target response 
        and a set of target features, marginalizing over the values of all other features (the complement 
        features). Intuitively, we can interpret the partial dependence as the expected target response 
        as a function of the target features.
        
        One-way PDPs tell us about the interaction between the target response and the target feature 
        (e.g. linear, non-linear). Note that PDPs **assume that the target features are independent** from 
        the complement features, and this assumption is often violated in practice.  If correlated 
        features can be reduced, these might be more meaningful.

        PDPs with two target features show the interactions among the two features.
        
        Notes
        -----
        See `sklearn.inspection.plot_partial_dependence`.
        
        Example
        -------
        >>> from sklearn.datasets import make_hastie_10_2
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from sklearn.inspection import plot_partial_dependence

        >>> X, y = make_hastie_10_2(random_state=0)
        >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X, y)
        >>> features = [0, 1, (0, 1)]
        >>> InspectModel.pdp(clf, X, features) 
        
        Parameters
        ----------
        model : BaseEstimator
            A fitted sklearn estimator.
        X : array-like, shape (n_samples, n_features)
            Dense grid used to build the grid of values on which the dependence will be evaluated. 
            **This is usually the training data.**
        features : list of {int, str, pair of int, pair of str}
            The target features for which to create the PDPs.
            If features[i] is an int or a string, a one-way PDP is created; if
            features[i] is a tuple, a two-way PDP is created. Each tuple must be
            of size 2.
        """
        from sklearn.inspection import plot_partial_dependence
        return plot_partial_dependence(model, X, features, **kwargs) 
    
    @staticmethod
    def pfi(model, X, y, n_repeats=30, feature_names=None, visualize=False):
        """
        Permutation feature importance is a model inspection technique that can be used for any 
        fitted estimator **when the data is tabular.** The permutation feature importance is defined 
        to be the decrease in a model score when a single feature value is randomly shuffled.
        It is indicative of **how much the model depends on the feature.**
        
        Can be computed on the training and/or test set (better).  There is some disagreement about which is 
        actually better.  Sklearn says that: "Permutation importances can be computed either on the 
        training set or on a held-out testing or validation set. Using a held-out set makes it possible 
        to highlight which features contribute the most to the **generalization power** of the inspected model. 
        Features that are important on the training set but not on the held-out set might cause the model 
        to overfit."
        
        **Features that are deemed of low importance for a bad model (low cross-validation score) could be 
        very important for a good model.**  The pfi is only important if the model itself is good.
        
        The sums of the pfi should roughly add up to the model's accuracy (or whatever score metric is used),
        if the features are independent, however, unlike Shapley values, this will not be exact. In other 
        words: results[results['95% CI > 0']]['Mean'].sum() / model.score(X_val, y_val) ~ 1.
        
        ``The importance measure automatically takes into account all interactions with other features. 
        By permuting the feature you also destroy the interaction effects with other features. This means that the 
        permutation feature importance takes into account both the main feature effect and the interaction effects 
        on model performance. This is also a disadvantage because the importance of the interaction between two 
        features is included in the importance measurements of both features. This means that the feature 
        importances do not add up to the total drop in performance, but the sum is larger. Only if there is no 
        interaction between the features, as in a linear model, the importances add up approximately.''
         - https://christophm.github.io/interpretable-ml-book/feature-importance.html
        
        For further advantages of pfi, see https://scikit-learn.org/stable/modules/permutation_importance.html.
        One of particular note is that pfi place too much emphasis on unrealistic inputs; this is because
        permuting features breaks correlations between features.  If you can remove those correlations
        (see Note below) then pfi's are more meaningful.
        
        Note
        ----
        When two features are correlated and one of the features is permuted, the model will still have 
        access to the feature through its correlated feature. This will result in a lower importance value 
        for both features, where they might actually be important.  One way to solve this is to cluster 
        correlated features and take only 1. **See `InspectData.cluster_collinear` for example.**
        """
        from sklearn.inspection import permutation_importance
        X = np.array(X)
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0)
        results = []
        naming = lambda i:feature_names[i] if feature_names != None else i
        for i in r.importances_mean.argsort()[::-1]:
            results.append([naming(i), r.importances_mean[i], r.importances_std[i], r.importances_mean[i]-2.0*r.importances_std[i] > 0])
        results = pd.DataFrame(data=results, columns=['Name or Index', 'Mean', 'Std', '95% CI > 0'])
        
        if visualize:
            perm_sorted_idx = r.importances_mean.argsort()
            plt.boxplot(r.importances[perm_sorted_idx].T, vert=False,
                        labels=feature_names[perm_sorted_idx])
            
        return results
    
    @staticmethod
    def kernelSHAP(model, X_train, X_test, use_probabilities=False, nsamples='auto', l1_reg=0.0, link='identity',
                  k_means=0):
        """
        Kernel SHAP (SHapley Additive exPlanations) is a way of estimating Shapley values using regression.
        
        Shapley values themselves are feature importances for linear models in the presence of multicollinearity.
        The importance of a feature is computed by adding to all subsets of coalitions of other features; when
        one or more collinear features are present, a features impact would be small, but there are lots of subsets
        where those collinear features are absent from the initial coalition, thus capturing (to some extent) this
        effect.
        
        As in PFI, it is best to try to remove correlated features first using hierarchical clustering; this will 
        make things easier computationally anyway.
        
        ``The fast computation makes it possible to compute the many Shapley values needed for the global 
        model interpretations. The global interpretation methods include feature importance, feature 
        dependence, interactions, clustering and summary plots. With SHAP, global interpretations are 
        consistent with the local explanations, since the Shapley values are the "atomic unit" of the 
        global interpretations. If you use LIME for local explanations and partial dependence plots plus 
        permutation feature importance for global explanations, you lack a common foundation.''
         - https://christophm.github.io/interpretable-ml-book/shap.html
        
        For notes and help interpreting the results, see:
        * https://github.com/slundberg/shap
        * https://christophm.github.io/interpretable-ml-book/shap.html
        
        Notes
        -----
        ``Shapley values are the only solution that satisfies properties of Efficiency, Symmetry, Dummy and 
        Additivity. SHAP also satisfies these, since it computes Shapley values.
        
        Be careful to interpret the Shapley value correctly: The Shapley value is the average contribution of a 
        feature value to the prediction in different coalitions. The Shapley value is NOT the difference in 
        prediction when we would remove the feature from the model.''
        
        - https://christophm.github.io/interpretable-ml-book/shap.html
        
        Examples
        --------
        # 1. A classification model with probabilities
        >>> import shap, sklearn
        >>> shap.initjs() # load JS visualization code to notebook
        >>> probability = True
        >>> link = 'logit' # Often better for models that return probabilities
        >>> X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
        >>> model = sklearn.svm.SVC(kernel='rbf', probability=probability)
        >>> model.fit(X_train, Y_train)
        >>> explainer, shap_values = InspectModel.kernelSHAP(model, X_train, X_test, use_probabilities=probability, link=link)
        >>> # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript, but JS is more interactive)
        >>> shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], # Just 1 instance for first (Setosa) class
        ...                 link=link, matplotlib=False)
        >>> shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link=link) # All instances for first (Setosa) class, rotated 90 degrees and stacked

        >>> # dependence_plot are not defined for probabilistic models
        >>> # summary_plot is always a bar plot if using a probabilistic model
        >>> shap.summary_plot(shap_values, X_test) 
        
        # 2. A regression model
        >>> import shap, sklearn
        >>> shap.initjs() # load JS visualization code to notebook
        >>> X,y = shap.datasets.boston()
        >>> model = sklearn.ensemble.RandomForestRegressor()
        >>> model.fit(X, y) 
        >>> explainer, shap_values = InspectModel.kernelSHAP(model, X, X, nsamples=100)
        >>> # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript, but JS is more interactive)
        >>> shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]) # Just 1 instance
        >>> shap.force_plot(explainer.expected_value, shap_values, X) # All instances, rotated 90 degrees and stacked
        >>> # create a dependence plot to show the effect of a single feature across the whole dataset
        >>> shap.dependence_plot("RM", shap_values, X) # also see pdi
        >>> # summarize the effects of all the features
        >>> shap.summary_plot(shap_values, X)
        >>> shap.summary_plot(shap_values, X, plot_type="bar")
        
        Parameters
        ----------
        model : BaseEstimator
            A fitted sklearn (or other supported) model, with a predict() and/or predict_proba() method 
            implemented.
        X_train : pandas.DataFrame or ndarray
            Data set model was trained on.  The explainer is fit using this.
        X_test : pandas.DataFrame or ndarray
            The explainer predicts SHAP values for these results.  In reality, you could provide X_train again
            here if you wanted to compute values for that set.
        use_probabilities : bool
            Use predict_proba() for model - this should only be used for classification tasks.
        nsamples : int or str
            Number of samples to use when computing shap values.  See ``shap.KernelExplainer.shap_values``.
        l1_reg : float
            Strength of l1 regularization to use computing shap values. See ``shap.KernelExplainer.shap_values``.
            Default of 0 does not do regularization since I'm not sure this computes valid Shapley values.
        link : str
            Link function to match feature importance values to the model output.  See ``shap.KernelExplainer``.
            Generally best to use 'logit' when use_probabilities=True, and 'identity' when use_probabilities=False;
            however, these examples do not. (1) https://slundberg.github.io/shap/notebooks/Iris%20classification%20with%20scikit-learn.html,
            (2)  https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20scikit-learn.html.
        k_means : int
            If > 0, use KMeans to summarize the training dataset which can greatly accelerate the calculation at the cost
            of accuracy.  This summarizes a dataset with k_means samples weighted by the number of data points they
            each represent.
        """
        import shap
        
        if (k_means > 0):
            X_train = shap.kmeans(X_train, k_means)
        
        explainer = shap.KernelExplainer(model=(model.predict_proba if use_probabilities else model.predict),
                                         data=X_train, 
                                         link=link
                                        )
        shap_values = explainer.shap_values(X_test, 
                                            nsamples=nsamples, 
                                            l1_reg=l1_reg, 
                                           )
        
        return explainer, shap_values
    
    @staticmethod
    def treeSHAP(model, X_train, X_test, approximate=False, check_additivity=True):
        """
        A specialized (faster) implementation of kernelSHAP for tree-based models that is EXACT, not an approximation,
        of the SHapley values.
        
        See Lundberg et al. "From local explanations to global understanding with explainable AI for trees" Nat. Mach. Intell. (2020)
        
        Example
        -------
        >>> import shap
        >>> shap.initjs() # load JS visualization code to notebook
        >>> X,y = shap.datasets.boston()
        >>> model = sklearn.ensemble.RandomForestRegressor()
        >>> model.fit(X, y) 
        >>> explainer, shap_values, interaction_values = InspectModel.treeSHAP(model, X, X)
        >>> # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript, but JS is more interactive)
        >>> shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]) # Just 1 instance
        >>> shap.force_plot(explainer.expected_value, shap_values, X) # All instances, rotated 90 degrees and stacked
        >>> # create a dependence plot to show the effect of a single feature across the whole dataset
        >>> shap.dependence_plot("RM", shap_values, X) # also see pdi
        >>> # summarize the effects of all the features
        >>> shap.summary_plot(shap_values, X)
        >>> shap.summary_plot(shap_values, X, plot_type="bar")

        Parameters
        ----------
        model : BaseEstimator
            A fitted sklearn (or other supported) model, with a predict() and/or predict_proba() method 
            implemented.
        X_train : pandas.DataFrame or ndarray
            Data set model was trained on.  The explainer is fit using this.
        X_test : pandas.DataFrame or ndarray
            The explainer predicts SHAP values for these results.  In reality, you could provide X_train again
            here if you wanted to compute values for that set.
        approximate : bool
            See ``shap.TreeExplainer.shap_values``.
        check_additivity : bool
            See ``shap.TreeExplainer.shap_values``.
        """
        import shap
        
        explainer = shap.TreeExplainer(model=model,
                                       data=X_train,
                                       feature_perturbation='tree_path_dependent' # shap_interaction_values only supported for this option at the moment
                                      )
        shap_values = explainer.shap_values(X_test, 
                                            check_additivity=check_additivity,
                                            approximate=approximate, 
                                           )
        
        interaction_values = explainer.shap_interaction_values(X_test)
        
        return explainer, shap_values, interaction_values
        
    @staticmethod
    def samplingSHAP(model, X_train, X_test, background=None, use_probabilities=False, nsamples='auto', l1_reg=0.0,
                  k_means=0):
        """ 
        Alternative to KernelShap.  From shap documentation: "This is an extension of the Shapley 
        sampling values explanation method (aka. IME) SamplingExplainer computes SHAP values under 
        the assumption of feature independence and is an extension of the algorithm proposed in 
        "An Efficient Explanation of Individual Classifications using Game Theory", Erik Strumbelj, 
        Igor Kononenko, JMLR 2010. It is a good alternative to KernelExplainer when you want to use 
        a large background set (as opposed to a single reference value for example)."
        
        It is important to note that this approximation method of Shapley values requires the assumption of 
        feature independence; furthermore, kernelSHAP is allegedly more computationally efficient. 
        
        - Lundberg & Lee "A unified approach to interpreting model predictions" NIPS (2017)
 
        See ``shap.SamplingExplainer`` for more details.
        
        Parameters
        ----------
        model : BaseEstimator
            A fitted sklearn (or other supported) model, with a predict() and/or predict_proba() method 
            implemented.
        X_train : pandas.DataFrame or ndarray
            Data set model was trained on.  The explainer is fit using this.
        X_test : pandas.DataFrame or ndarray
            The explainer predicts SHAP values for these results.  In reality, you could provide X_train again
            here if you wanted to compute values for that set.
        background : pandas.DataFrame or ndarray
            From shap documentation: ``The background dataset to use for integrating out features. To determine the impact
            of a feature, that feature is set to "missing" and the change in the model output
            is observed. Since most models aren't designed to handle arbitrary missing data at test
            time, we simulate "missing" by replacing the feature with the values it takes in the
            background dataset. So if the background dataset is a simple sample of all zeros, then
            we would approximate a feature being missing by setting it to zero. Unlike the
            KernelExplainer this data can be the whole training set, even if that is a large set. This
            is because SamplingExplainer only samples from this background dataset.''
            If set to None (default) this uses X as the background also.
        use_probabilities : bool
            Use predict_proba() for model - this should only be used for classification tasks.
        nsamples : int or str
            Number of samples to use when computing shap values.  See ``shap.KernelExplainer.shap_values``.
        l1_reg : float
            Strength of l1 regularization to use computing shap values. See ``shap.KernelExplainer.shap_values``.
            Default of 0 does not do regularization since I'm not sure this computes valid Shapley values.
        k_means : int
            If > 0, use KMeans to summarize the dataset which can greatly accelerate the calculation at the cost
            of accuracy.  This summarizes a dataset with k_means samples weighted by the number of data points they
            each represent.
        """
        import shap
        
        if (k_means > 0):
            X_train = shap.kmeans(X_train, k_means)
            
        if (background is None):
            background = X_train
        
        explainer = shap.SamplingExplainer(model=(model.predict_proba if use_probabilities else model.predict),
                                         data=background
                                        )
        shap_values = explainer.shap_values(X_test, 
                                            nsamples=nsamples, 
                                            l1_reg=l1_reg, 
                                           )
        
        return explainer, shap_values
        
    @staticmethod
    def deepSHAP():
        """ Deep Neural Nets """
        raise NotImplementedError
        
    @staticmethod
    def LIME():
        # https://github.com/marcotcr/lime
        # https://christophm.github.io/interpretable-ml-book/lime.html
        raise NotImplementedError
        
