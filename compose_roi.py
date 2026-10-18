"""Combine two compatible semantic region checks without new inference.

Each contrast keeps its own distinct-frame consensus. This creates an offline
candidate; it does not register the candidate in the player.
"""
import argparse
import copy
from pathlib import Path
from labeling import read_json,write_json
from provenance import file_hash
from roi_gate import validate_head


def compose(first,second):
    for head in (first,second):
        validate_head(head)
        if (head.get('feature_kind','clip')!='clip' or head.get('head_type')!='semantic-prototype'
                or head.get('negative_confirmations') not in (3,5)
                or not head.get('clip_identity') or not head.get('anatomy_identity')
                or any(k in head for k in ('secondary_head','detector_head'))):
            raise ValueError('Compose two independently calibrated semantic checks')
    result=copy.deepcopy(first);result['secondary_head']=copy.deepcopy(second)
    validate_head(result)
    result['selection_warning']='Two semantic contrasts share embeddings but require separate frame consensus; complete interval comparison required'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True);parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new candidate name')
    result=compose(read_json(args.first),read_json(args.second))
    result['component_identities']=[file_hash(args.first),file_hash(args.second)]
    write_json(args.output,result)
