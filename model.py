import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Resources:
# https://christophm.github.io/interpretable-ml-book/

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
    def kernelSHAP(model, X, use_probabilities=False, nsamples='auto', l1_reg=0.0, link='identity'):
        """
        SHAP (SHapley Additive exPlanations) is a way of estimating Shapley values.
        
        Shapley values themselves are feature importances for linear models in the presence of multicollinearity.
        However, it seems multicollinearity is still a problem.  TreeSHAP gets around this, but has issues with
        sometimes causing unintuitive results.
        
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
        prediction when we would remove the feature from the model.
        
        Like many other permutation-based interpretation methods, the Shapley value method suffers from inclusion 
        of unrealistic data instances when features are correlated. To simulate that a feature value is missing 
        from a coalition, we marginalize the feature. This is achieved by sampling values from the feature's 
        marginal distribution. This is fine as long as the features are independent. When features are dependent, 
        then we might sample feature values that do not make sense for this instance. But we would use those to 
        compute the feature's Shapley value. To the best of my knowledge, there is no research on what that means 
        for the Shapley values, nor a suggestion on how to fix it. One solution might be to permute correlated 
        features together and get one mutual Shapley value for them. Or the sampling procedure might have to be 
        adjusted to account for dependence of features.
        
        KernelSHAP therefore suffers from the same problem as all permutation-based interpretation methods. The 
        estimation puts too much weight on unlikely instances. Results can become unreliable. But it is necessary 
        to sample from the marginal distribution. If the absent feature values would be sampled from the 
        conditional distribution, then the resulting values are no longer Shapley values. The resulting values 
        would violate the Shapley axiom of Dummy, which says that a feature that does not contribute to the outcome 
        should have a Shapley value of zero.''
        
        - https://christophm.github.io/interpretable-ml-book/shap.html
        
        Examples
        --------
        # 1. A classification model with probabilities
        >>> import shap, sklearn
        >>> shap.initjs() # load JS visualization code to notebook
        >>> probability = True
        >>> link = 'logit' # Better for models that return probabilities
        >>> X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
        >>> model = sklearn.svm.SVC(kernel='rbf', probability=probability)
        >>> model.fit(X_train, Y_train)
        >>> explainer, shap_values = InspectModel.kernelSHAP(model, X_test, use_probabilities=probability, link=link)
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
        >>> explainer, shap_values = InspectModel.kernelSHAP(model, X, nsamples=100)
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
        X : pandas.DataFrame or ndarray
            Generally X_test is advised.  Use only if X.shape[1] < 1000 to complete in a < 1 hr if < 20 features.
            Using less nsamples can also accelerate instead of using smaller X - shap.SamplingExplainer is 
            another alternative altogether.
        use_probabilities : bool
            Use predict_proba() for model - this should only be used for classification tasks.
        nsamples : int or str
            Number of samples to use when computing shap values.  See ``shap.KernelExplainer.shap_values``.
        l1_reg : float
            Strength of l1 regularization to use computing shap values. See ``shap.KernelExplainer.shap_values``.
            Default of 0 does not do regularization since I'm not sure this computes valid Shapley values.
        link : str
            Link function to match feature importance values to the model output.  See ``shap.KernelExplainer``.
            Best to use 'logit' when use_probabilities=True, and 'identity' when use_probabilities=False.
        """
        import shap
        
        explainer = shap.KernelExplainer(model=(model.predict_proba if use_probabilities else model.predict),
                                         data=X, 
                                         link=link
                                        )
        shap_values = explainer.shap_values(X, 
                                            nsamples=nsamples, 
                                            l1_reg=l1_reg, 
                                           )
        
        return explainer, shap_values
    
    @staticmethod
    def treeSHAP(model, X, approximate=False, check_additivity=True):
        """
        A specialized (faster) implementation of kernelSHAP for tree-based models.
        
        Example
        -------
        >>> import shap
        >>> shap.initjs() # load JS visualization code to notebook
        >>> X,y = shap.datasets.boston()
        >>> model = sklearn.ensemble.RandomForestRegressor()
        >>> model.fit(X, y) 
        >>> explainer, shap_values, interaction_values = InspectModel.treeSHAP(model, X)
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
        X : pandas.DataFrame or ndarray
            Generally X_test is advised.  
        approximate : bool
            See ``shap.TreeExplainer.shap_values``.
        check_additivity : bool
            See ``shap.TreeExplainer.shap_values``.
        """
        import shap
        
        explainer = shap.TreeExplainer(model=model,
                                       data=X,
                                       feature_perturbation='tree_path_dependent' # shap_interaction_values only supported for this option at the moment
                                      )
        shap_values = explainer.shap_values(X, 
                                            check_additivity=check_additivity,
                                            approximate=approximate, 
                                           )
        
        interaction_values = explainer.shap_interaction_values(X)
        
        return explainer, shap_values, interaction_values
        
    @staticmethod
    def samplingSHAP():
        """ Alternative to KernelShap """
        raise NotImplementedError
        
    @staticmethod
    def deepSHAP():
        """ Deep Neural Nets """
        raise NotImplementedError
        
    @staticmethod
    def LIME():
        # https://github.com/marcotcr/lime
        # https://christophm.github.io/interpretable-ml-book/lime.html
        raise NotImplementedError
        