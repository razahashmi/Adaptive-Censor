"""Pixel coverage and clean-pixel disturbance against independent binary masks.

PNG masks must share a frame size. White pixels are positive/reviewed; black
pixels are negative/unreviewed. These measure annotated mask geometry, not the
semantic accuracy of unannotated pictures or how much detail blur reveals.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from labeling import read_json, write_json


def counts(predicted, gold, reviewed=None):
    predicted, gold = np.asarray(predicted,dtype=bool), np.asarray(gold,dtype=bool)
    if predicted.shape!=gold.shape or predicted.ndim!=2:
        raise ValueError('Masks must have the same two-dimensional shape')
    reviewed=np.ones_like(gold) if reviewed is None else np.asarray(reviewed,dtype=bool)
    if reviewed.shape!=gold.shape: raise ValueError('Review mask shape differs')
    return dict(tp=int((predicted & gold & reviewed).sum()),fp=int((predicted & ~gold & reviewed).sum()),
                fn=int((~predicted & gold & reviewed).sum()),clean=int((~gold & reviewed).sum()),
                gold=int((gold & reviewed).sum()))


def rates(c):
    divide=lambda a,b:a/b if b else None
    return dict(coverage=divide(c['tp'],c['gold']), clean_pixel_disturbance=divide(c['fp'],c['clean']),
                iou=divide(c['tp'],c['tp']+c['fp']+c['fn']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=read_json(args.manifest);total=dict(tp=0,fp=0,fn=0,clean=0,gold=0);rows=[]
    for sample in data['samples']:
        if sample.get('labeler')!='human':raise ValueError('Pixel ground truth requires independent human labels')
        def load(name):
            return np.asarray(Image.open(args.manifest.parent/sample[name]).convert('L'))>=128
        c=counts(load('predicted_mask'),load('gold_mask'),load('reviewed_mask') if sample.get('reviewed_mask') else None)
        for key,value in c.items():total[key]+=value
        rows.append(dict(id=sample['id'],counts=c,**rates(c)))
    write_json(args.output,dict(samples=rows,counts=total,**rates(total)))


if __name__=='__main__':main()
