"""Calibrate an experimental semantic anatomy veto from reviewed crops.

Uses frozen CLIP text directions, not fitted classifier weights. The threshold
uses training labels; it is not a held-out accuracy estimate or qualification.
"""
import argparse
from pathlib import Path
import numpy as np
from analysis_engine import Models
from labeling import read_json,write_json
from learned import thresholds
from model_setup import prepare_environment
from provenance import digest,file_hash
from select_windows import normalize_title

POSITIVE=("A close-up photo of a woman's exposed breast and nipple.",
          'A woman with bare breasts and visible nipples.',
          'A topless woman with exposed breasts.',
          'A close-up photo of a bare human buttocks.',
          'A person with exposed bare buttocks.')
NEGATIVE=("A close-up photo of a shirtless man's chest and nipple.",
          'A man with a bare muscular chest.',
          'A shirtless man with exposed pectoral muscles.',
          'A male torso with a bare chest and male nipples.')
COVERED_POSITIVE=("A close-up photo of a woman's exposed breast with a visible bare nipple.",
                  'A topless woman with bare breasts and exposed nipples.',
                  'An uncovered female breast with a visible nipple and areola.')
COVERED_NEGATIVE=('A woman wearing a bra covering her breasts and nipples.',
                  'A woman wearing a swimsuit with her breasts covered.',
                  'A woman with cleavage but her nipples concealed by clothing.',
                  'A close-up of a clothed female chest with no nipple visible.',
                  'A close-up photo of a bare shoulder or arm.',
                  'A close-up photo of a human face.',
                  'A close-up photo of a shirtless man with a male chest.')


def calibrate(path,output,models,allow_ai_labels=False,contrast='male-chest'):
    data=read_json(path)
    if file_hash(path.parent/'features.npz')!=data['feature_identity']:
        raise ValueError('ROI features changed')
    if data['clip_identity']!=file_hash(models.root/'clip/onnx/vision_model_quantized.onnx'):
        raise ValueError('ROI encoder changed')
    if not data.get('crop_method','').lower().startswith('raw detector box'):
        raise ValueError('Prepare raw detector crops matching runtime, rather than legacy segment masks')
    selected=[(i,r) for i,r in enumerate(data['rows']) if r.get('split')=='train' and r.get('label') in (0,1)
              and r.get('detector_class')=='FEMALE_BREAST_EXPOSED'
              and (r.get('labeler')=='human' or (allow_ai_labels and r.get('labeler') in ('codex','claude')))]
    if len(selected)<40:
        raise ValueError('Review at least 40 breast-class crops; AI labels require explicit opt-in')
    y=np.array([r['label'] for _,r in selected]);groups=np.array([normalize_title(r['title']) for _,r in selected])
    if len(set(groups))<5 or any(len(set(groups[y==v]))<3 for v in (0,1)):
        raise ValueError('Need at least 5 titles and each class in at least 3 titles')
    views=data.get('feature_views',['local'])
    if views not in (['local'],['body'],['local','body']):
        raise ValueError('Unknown or reordered feature views')
    with np.load(path.parent/'features.npz') as archive:
        X=archive['embeddings'][[i for i,_ in selected]].astype(np.float32)
    if X.shape!=(len(y),512*len(views)) or not np.isfinite(X).all():
        raise ValueError('Invalid ROI feature matrix')
    if contrast not in ('male-chest','covered-anatomy'):raise ValueError('Unknown semantic contrast')
    positive,negative_prompts=(POSITIVE,NEGATIVE) if contrast=='male-chest' else (COVERED_POSITIVE,COVERED_NEGATIVE)
    vectors=models.text_embeddings(list(positive+negative_prompts),'cpu')
    direction=vectors[:len(positive)].mean(axis=0)-vectors[len(positive):].mean(axis=0)
    coef=np.concatenate([direction*.75,direction*.25]) if len(views)==2 else direction
    scores=X@(coef.astype(np.float64) if contrast=='covered-anatomy' else coef)
    select_threshold=lambda values:float(values[y==1].min())-1e-6 if contrast=='covered-anatomy' else thresholds(values,y)[0]
    threshold=select_threshold(scores)
    weights={'local':.75,'body':.25} if len(views)==2 else {views[0]:1.}
    view_thresholds={view:select_threshold(X[:,column*512:(column+1)*512]@direction) for column,view in enumerate(views)}
    negative=scores<threshold
    for column,view in enumerate(views):negative &= X[:,column*512:(column+1)*512]@direction<view_thresholds[view]
    head=dict(mean=np.zeros(len(coef)).tolist(),scale=np.ones(len(coef)).tolist(),coef=coef.tolist(),intercept=0.,
        decision_threshold=threshold,feature_views=views,head_type='semantic-prototype',
        clip_identity=data['clip_identity'],dataset_identity=file_hash(path),feature_identity=data['feature_identity'],
        supported_classes=['FEMALE_BREAST_EXPOSED'],crops=len(y),titles=sorted(set(groups)),
        class_positive_titles={'FEMALE_BREAST_EXPOSED':sorted(set(groups[y==1]))},
        experimental=True,ai_labels_allowed=allow_ai_labels,
        labelers={name:sum(r['labeler']==name for _,r in selected) for name in sorted({r['labeler'] for _,r in selected})},
        operating_point='training-recall',training_recall=float((~negative&(y==1)).sum()/sum(y==1)),
        training_negative_rejection=float((negative&(y==0)).sum()/sum(y==0)),
        negative_confirmations=3,view_weights=weights,view_thresholds=view_thresholds,
        contrast=contrast,
        prototypes=dict(positive=list(positive),negative=list(negative_prompts),identity=digest([list(positive),list(negative_prompts)])),
        selection_warning='Training-only threshold; independent precision/recall and mask evaluation required')
    for key in ('anatomy_model','anatomy_identity'):
        if key in data:head[key]=data[key]
    if output.exists():raise FileExistsError('Choose a new model name')
    write_json(output,head)
    return head


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True);parser.add_argument('--model',required=True)
    parser.add_argument('--allow-ai-labels',action='store_true')
    parser.add_argument('--contrast',choices=['male-chest','covered-anatomy'],default='male-chest')
    args=parser.parse_args()
    if not args.model or len(args.model)>64 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.model):
        parser.error('Use a lowercase model name containing letters, digits and hyphens')
    root=prepare_environment();models=Models()
    try:
        head=calibrate(args.labels,root/'roi-models'/f'{args.model}.json',models,args.allow_ai_labels,args.contrast)
        print({k:head[k] for k in ('crops','titles','training_recall','training_negative_rejection','experimental')})
    finally:models.close()


if __name__=='__main__':main()
