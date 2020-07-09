import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class InspectData:
    def __init__(self):
        pass
    
    @staticmethod
    def cluster_elbow(X, clusters=range(1, 11)):
        """
        Uses kmeans++ to examine within-cluster sum squared errors (inertia or distortion)
        and plots as the number of clusters increases.  Because of the use of
        kmeans, this is best if the clusters are more spherical.

        See Ch. 11 of "Python Machine Learning" by Raschka & Mirjalili.
        """
        from sklearn.cluster import KMeans

        distortions = []
        for i in clusters:
            km = KMeans(n_clusters=i, 
                        init='k-means++', 
                        n_init=10, 
                        max_iter=300, 
                        random_state=0)
            km.fit(X)
            distortions.append(km.inertia_)
        plt.plot(clusters, distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.tight_layout()

        return plt.gca()
    
    @staticmethod
    def cluster_silhouette(X, y):
        """
        Plot silhouette curves to assess the quality of clustering into a
        meaningful number of clusters in **classification tasks**. Ideal silhouette 
        coefficients are close to 1, meaning "tight" well-separated clusters.
        
        **Because this is supervised this could introduce bias.  Be careful.**

        See Ch. 11 of "Python Machine Learning" by Raschka & Mirjalili.
        """
        from matplotlib import cm
        from sklearn.metrics import silhouette_samples

        cluster_labels = np.unique(y)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, y, metric='euclidean')

        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                    edgecolor='none', color=color)

            yticks.append((y_ax_lower + y_ax_upper) / 2.)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--") 

        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette coefficient')

        plt.tight_layout()

        return plt.gca()
    
    @staticmethod
    def cluster_collinear(X, feature_names=None, figsize=None, t=None, display=True):
        """
        Use Ward clustering to cluster collinear features and select a single feature from each cluster.
        See https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

        This can be used as a preprocessing step since it is unsupervised.
        
        Example
        -------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        >>> # Train model in first round
        >>> clf = RandomForestClassifier(n_estimators=100, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> clf.score(X_test, y_test) # 97%

        >>> # Look at pfi --> EVERYTHING comes out as irrelevant because many features highly correlated
        >>> InspectModel.pfi(clf, X_test, y_test, n_repeats=30, feature_names=data.feature_names.tolist())

        >>> # Look at multicollinearity
        >>> selected_features, cluster_id_to_feature_ids = InspectModel.cluster_collinear(X, # Can use entire dataset since this is unsupervised
        ...                                                                                 figsize=(12, 8), 
        ...                                                                                 display=True, 
        ...                                                                                 t=2,
        ...                                                                                 feature_names=None) # Get indices to work with

        >>> # Fit again just using these selected features
        >>> X_train, X_test = X_train[:,selected_features], X_test[:,selected_features]
        >>> clf.fit(X_train, y_train) 
        >>> clf.score(X_test, y_test) # 96%, almost identical as expected

        >>> # Top is 'mean radius', which according to dendogram above, is highly correlated with other "size" metrics
        >>> InspectModel.pfi(clf, X_test, y_test, n_repeats=30, feature_names=data.feature_names[selected_features])

        Notes
        -----
        If feature names are provided, names are returned.  Otherwise they are the indices of the 
        columns in X.

        Parameters
        ---------
        X : array-like
            Dense feature matrix.
        feature_names : list or None
            Names of each column in X (in ascending order). If None, returns indices of columns,
            otherwise all outputs are in terms of names.
        figsize : tuple or None
            Size of visualization to display.  Ignored if display = False.
        t :  float or None
            Ward clustering threshold to determine the number of clusters.
        display : bool
            Whether or not to visualize results.

        Returns
        -------
        selected_features, cluster_id_to_feature_ids
        """
        from collections import defaultdict
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy

        X = np.array(X)
        if feature_names is None:
            naming = lambda i:i
        else:
            feature_names = list(feature_names) # Needs to be a list for compatibility elsewhere
            naming = lambda i:feature_names[i] 

        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)

        # If no specification for where to cut, guess
        guess = (np.sqrt(np.max(corr_linkage))/3.) if t == None else t

        cluster_ids = hierarchy.fcluster(corr_linkage, t=guess, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(naming(idx))

        # Arbitrarily select the first feature put into each cluster
        selected_features = np.array([v[0] for v in cluster_id_to_feature_ids.values()])

        # Plot
        if (display):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            ax1.axhline(guess, color='k')
            decorate = lambda x: '***'+str(x).upper()+'***'
            if feature_names:
                labels = list(feature_names)
            else:
                labels = np.arange(X.shape[1]).tolist()
            for i in range(len(labels)):
                if labels[i] in selected_features:
                    labels[i] = decorate(labels[i])

            dendro = hierarchy.dendrogram(corr_linkage, ax=ax1, labels=labels,
                                          leaf_rotation=90, color_threshold=guess)

            dendro_idx = np.arange(0, len(dendro['ivl']))
            ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
            _ = ax2.set_xticks(dendro_idx)
            _ = ax2.set_yticks(dendro_idx)
            _ = ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
            _ = ax2.set_yticklabels(dendro['ivl'])
            _ = ax2.set_title('Spearman Rank-Order Correlations')

            fig.tight_layout()

        return selected_features, cluster_id_to_feature_ids
        
    @staticmethod
    def pairplot(df, **kwargs):
        """
        A pairplot of the data.
        See https://seaborn.pydata.org/generated/seaborn.pairplot.html.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame with (dense) X predictors.  It may or may not contain a column
            for the predictin target.  For classification tasks, this can be visualized
            using 'hue' as shown below.
        
        Example
        -------
        >>> from sklearn.datasets import load_breast_cancer
        >>> data = load_breast_cancer()
        >>> df = pd.DataFrame(data=data.data, columns=data.feature_names)
        >>> df['target'] = data.target
        >>> InspectData.pairplot(df, vars=df.columns[0:5], hue='target', diag_kind='kde')
        >>> InspectData.pairplot(df, vars=df.columns[0:5], hue='target', diag_kind='auto')
        """
        sns.pairplot(df, **kwargs)