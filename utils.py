import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.signal import argrelextrema
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from numpy import mean, array, linspace
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN


__all__ = ['plotCategory', 'getMajorityClass', 'getClassWeight', 'getSampleDict', 'getResamplePipeline', 'crossValidate',
           'randomForestFeatureImportancePlot', 'permuationFeatureImportancePlot', 'clusterDBSCAN', 'clusterKDE', 'correlationPlot']



def plotCategory(df, field):
    items = df[field].value_counts()
    plt.figure(figsize=(15, 5))
    plt.bar(items.keys(), items.values)
    plt.xticks(rotation=45)
    plt.show()


def getMajorityClass(y):
    d = Counter(y)
    max_ = -1
    majority = None
    for key, val in d.items():
        if val > max_:
            majority = key
            max_ = val
    return majority


def getSampleDict(y, overSampleRate=0.5):
    labelDict = Counter(y)
    maj = getMajorityClass(y)
    majorityCount = labelDict[maj]
    sampleDict = {key: int(overSampleRate * majorityCount)
                  for key in labelDict.keys() if key != maj}
    return sampleDict


def getClassWeight(y, maj):
    labelDict = Counter(y)
    classWeight = {key: 2 for key in labelDict.keys() if key != maj}
    classWeight[maj] = 1
    return classWeight


def getResamplePipeline(sampleDict, k_neighbors=3, withUnder=True):
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    over = SMOTE(sampling_strategy=sampleDict,
                 random_state=42, k_neighbors=k_neighbors)
    steps = [('encoding', enc), ('over', over)]
    if withUnder:
        under = RandomUnderSampler(sampling_strategy='majority')
        steps.append(('under', under))
    pipeline = Pipeline(steps=steps)
    return pipeline


def crossValidate(model, X, y, n_splits=3, n_repeats=3):
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    return mean(scores)


def randomForestFeatureImportancePlot(X, y, feature_names, model, isPipeline=False):
    model.fit(X, y)
    if isPipeline:
        model = model.named_steps['model']
    tree_feature_importances = (model.feature_importances_)
    sorted_idx = tree_feature_importances.argsort()

    y_ticks = np.arange(0, len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, tree_feature_importances[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("Random Forest Feature Importances (MDI)")
    fig.set_size_inches(18.5, 15.5)
    plt.tick_params(axis='y', which='major', labelsize=8)
    fig.tight_layout()
    plt.show()


def permuationFeatureImportancePlot(model, X, y, feature_names, n_repeats=10, plot=True):
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feature_names[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.set_size_inches(18.5, 15.5)
    plt.tick_params(axis='y', which='major', labelsize=8)
    fig.tight_layout()
    plt.show()
    return result


def clusterKDE(labels, plot=False):
    targets = labels.to_numpy().reshape(-1, 1)
    labels_ = np.empty(len(targets))
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(targets)
    space = linspace(0, max(targets) + 1)
    e = kde.score_samples(space.reshape(-1, 1))
    if plot:
        plt.plot(space, e)
        plt.show()
    localMinimum = argrelextrema(e, np.less)[0]
    for idx, item in enumerate(targets):
        val = item[0]
        cls = 0
        while True:
            if cls >= len(localMinimum) or val < localMinimum[cls]:
                labels_[idx] = cls
                break
            cls += 1
    return labels_


def clusterDBSCAN(labels, eps=4, min_samples=5):
    targets = labels.to_numpy().reshape(-1, 1)
    clustering = DBSCAN(eps=4, min_samples=5).fit(targets)
    return clustering.labels_


def correlationPlot(X, labels, top=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(np.nan_to_num(corr))
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()
