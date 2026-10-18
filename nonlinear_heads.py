"""Experimental compact tree heads over frozen video/image/audio features.

Preprocessing and operating points are fitted with whole titles held out.
This command never qualifies a model for automatic filtering.
"""
import argparse
from pathlib import Path
import time

import numpy as np


def project_inputs(projection, matrix):
    values=(np.asarray(matrix,dtype=np.float64)-np.asarray(projection['mean']))/np.asarray(projection['scale'])
    components=np.asarray(projection['components'])
    return values @ components.T - np.asarray(projection['center']) @ components.T


def tree_scores(head, matrix):
    values=np.asarray(matrix,dtype=np.float64)
    scores=np.full(len(values),head['intercept'],dtype=np.float64)
    rows=np.arange(len(values))
    for tree in head['trees']:
        feature=np.asarray(tree['feature'],dtype=np.int32)
        threshold=np.asarray(tree['threshold'])
        left,right=np.asarray(tree['left']),np.asarray(tree['right'])
        leaf=np.asarray(tree['leaf'],dtype=bool)
        nodes=np.zeros(len(values),dtype=np.int32)
        while not leaf[nodes].all():
            active=~leaf[nodes]
            indices=nodes[active]
            goes_left=values[rows[active],feature[indices]]<=threshold[indices]
            nodes[active]=np.where(goes_left,left[indices],right[indices])
        scores+=np.asarray(tree['value'])[nodes]
    return scores


def export_tree(model):
    trees=[]
    for predictors in model._predictors:
        nodes=predictors[0].nodes
        trees.append(dict(feature=nodes['feature_idx'].tolist(),threshold=nodes['num_threshold'].tolist(),
                          left=nodes['left'].tolist(),right=nodes['right'].tolist(),leaf=nodes['is_leaf'].tolist(),
                          value=nodes['value'].tolist()))
    return dict(model_type='histogram-trees',trees=trees,intercept=float(model._baseline_prediction[0,0]))


def train(folders,components=96,allow_legacy=False,log=print):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from labeling import LABEL_CATEGORIES,is_complete,read_json
    from learned import window_matrix,grid_labels,thresholds
    from select_windows import normalize_title
    from provenance import FEATURE_SCHEMA
    rows=[]
    kinds=['clip','vjepa','ast']
    for folder in folders:
        for window in read_json(folder/'project.json')['windows']:
            labels=read_json(folder/'labels'/f"{window['id']}.json",{})
            if window['split']!='train' or not is_complete(labels):continue
            times,matrix=window_matrix(folder,window,kinds,allow_legacy)
            target=np.stack([grid_labels(window,labels,c,times) for c in LABEL_CATEGORIES],1)
            keep=(target>=0).any(1)
            rows.append((window,matrix[keep],target[keep]))
    if not rows:raise ValueError('No complete training windows')
    X=np.concatenate([matrix for _,matrix,_ in rows]).astype(np.float64)
    y=np.concatenate([target for _,_,target in rows])
    groups=np.concatenate([[normalize_title(w['title'])]*len(matrix) for w,matrix,_ in rows])
    profiles=(7,15)
    oof={leaves:np.zeros(y.shape) for leaves in profiles}
    positive_titles=[len(set(groups[y[:,k]==1])) for k in range(len(LABEL_CATEGORIES))]
    def classifier(leaves):
        return HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=120,learning_rate=.05,
                    min_samples_leaf=20,l2_regularization=10.,class_weight='balanced',early_stopping=False,random_state=7)
    with threadpool_limits(limits=4):
        for fold,(fit,held) in enumerate(GroupKFold(5).split(X,groups=groups),1):
            started=time.perf_counter()
            scaler=StandardScaler().fit(X[fit])
            pca=PCA(n_components=min(components,len(fit),X.shape[1]),svd_solver='randomized',random_state=7).fit(scaler.transform(X[fit]))
            fit_X,held_X=pca.transform(scaler.transform(X[fit])),pca.transform(scaler.transform(X[held]))
            for leaves in profiles:
                for k,category in enumerate(LABEL_CATEGORIES):
                    known=y[fit,k]>=0
                    if positive_titles[k]<3 or len(np.unique(y[fit,k][known]))<2:
                        oof[leaves][held,k]=-100
                        continue
                    model=classifier(leaves).fit(fit_X[known],y[fit,k][known])
                    oof[leaves][held,k]=model.decision_function(held_X)
            log(f'fold {fold}/5: {time.perf_counter()-started:.1f}s',flush=True)
        scaler=StandardScaler().fit(X)
        pca=PCA(n_components=min(components,len(X),X.shape[1]),svd_solver='randomized',random_state=7).fit(scaler.transform(X))
        projection=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),center=pca.mean_.tolist(),components=pca.components_.tolist())
        transformed=project_inputs(projection,X)
        assert np.allclose(transformed,pca.transform(scaler.transform(X)),atol=1e-10,rtol=0)
        heads={}
        for k,category in enumerate(LABEL_CATEGORIES):
            known=y[:,k]>=0
            if positive_titles[k]<3:
                heads[category]=dict(skipped='Need positive examples from at least three training titles')
                continue
            results={leaves:float(average_precision_score(y[known,k],scores[known,k])) for leaves,scores in oof.items()}
            leaves=max(results,key=results.get)
            candidate,decision=thresholds(oof[leaves][known,k],y[known,k])
            model=classifier(leaves).fit(transformed[known],y[known,k])
            head=export_tree(model)
            assert np.allclose(tree_scores(head,transformed),model.decision_function(transformed),atol=1e-10,rtol=0)
            head.update(candidate_threshold=candidate,decision_threshold=decision,positive_titles=positive_titles[k],
                        positives=int((y[:,k]==1).sum()),cv_average_precision=results[leaves],cv_profiles=results,
                        max_leaf_nodes=leaves,iterations=120)
            heads[category]=head
            log(f'{category}: OOF AP {results[leaves]:.3f}, decision {decision}',flush=True)
    return dict(kinds=kinds,grid=.25,feature_schema=FEATURE_SCHEMA,projection=projection,heads=heads,
                windows=len(rows),titles=sorted({w['title'] for w,_,_ in rows}),experimental=True,
                legacy_features=allow_legacy,training_algorithm='PCA + title-held-out histogram boosting')


def main():
    from labeling import project_dir,write_json
    from model_setup import prepare_environment
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects',nargs='+',required=True)
    parser.add_argument('--heads',required=True)
    parser.add_argument('--components',type=int,default=96)
    parser.add_argument('--allow-legacy-features',action='store_true')
    args=parser.parse_args();root=prepare_environment();output=root/'heads'/f'{args.heads}.json'
    if output.exists():raise FileExistsError('Choose a new head name')
    result=train([project_dir(root,name) for name in args.projects],args.components,args.allow_legacy_features)
    write_json(output,result)
    print(output)


if __name__=='__main__':main()
