"""Failure reports must conserve missed time and reject stale benchmark inputs."""
import copy
from pathlib import Path
import tempfile
import unittest

from diagnose_failures import diagnose_window, load_frozen, report
from evaluate_labels import score_window
from labeling import LABEL_CATEGORIES, write_json
from provenance import digest, file_hash


def fixture():
    window = dict(id='clip', title='A', split='dev', target=[0,10], clip_duration=10)
    labels = dict(labeler='human', drafts_shown=False, categories={
        c:dict(status='reviewed', intervals=[[1,9]] if c=='nudity' else [], ambiguous=[])
        for c in LABEL_CATEGORIES})
    events = [dict(start=1,end=2,category='nudity'),
              dict(start=3,end=5,category='nudity',review_required=True),
              dict(start=9,end=9.5,category='nudity')]
    trace = [dict(start=4,end=7,category='nudity',stage='anatomy_raw',score=.8),
             dict(start=2.5,end=6,category='nudity',stage='anatomy_raw',score=.3)]
    return window, labels, dict(window='clip',events=events,trace=trace)


class FailureDiagnosisTests(unittest.TestCase):
    def test_overlapping_review_and_raw_spans_partition_actual_misses(self):
        w,l,p = fixture()
        result = diagnose_window(w,l,p,'nudity',False)
        self.assertEqual(result['caught_seconds'],1)
        self.assertEqual(result['missed_seconds'],7)
        self.assertEqual(result['false_seconds'],.5)
        self.assertEqual({k:v['seconds'] for k,v in result['miss_buckets'].items()},
                         dict(reviewable=2, proposed_without_review=2.5, no_recorded_support=2.5))
        self.assertEqual({k:v['seconds'] for k,v in result['raw_proposal_overlap'].items()},
                         dict(score_at_least_055=3, lower_score=1.5, no_raw_proposal=2.5))
        counts = score_window(w,l,p,'nudity',False)
        self.assertEqual((result['caught_seconds'],result['missed_seconds'],result['false_seconds']),
                         (counts['tp'],counts['fn'],counts['fp']))

    def test_ambiguous_spans_and_context_do_not_enter_failure_totals(self):
        w,l,p = fixture()
        w['target']=[2,8]
        l['categories']['nudity']['ambiguous']=[[4,6]]
        result = diagnose_window(w,l,p,'nudity',False)
        self.assertEqual(result['scored_seconds'],4)
        self.assertEqual(result['missed_seconds'],4)
        self.assertEqual(sum(v['seconds'] for v in result['miss_buckets'].values()),4)
        self.assertEqual(result['false_seconds'],0)

    def test_review_padding_is_accounted_without_claiming_raw_support(self):
        w,l,p = fixture()
        p.update(events=[dict(start=.7,end=.9,category='nudity',review_required=True)],trace=[])
        result = diagnose_window(w,l,p,'nudity',True)
        self.assertGreater(result['miss_buckets']['reviewable']['seconds'],0)
        self.assertEqual(result['miss_buckets']['proposed_without_review']['seconds'],0)
        self.assertEqual(result['raw_proposal_overlap']['no_raw_proposal']['seconds'],8)

    def test_unknown_categories_are_excluded_and_ai_labels_remain_explicit(self):
        w,l,p = fixture()
        l['labeler']='claude'
        l['categories']['nudity']=dict(status='unsure', intervals=[], ambiguous=[])
        result=report([('dev',w,l,p)],{},['nudity'],False)
        self.assertEqual(result['categories']['nudity']['unsure_windows'],1)
        self.assertEqual(result['windows_detail'][0]['categories'],{})
        self.assertEqual(result['labelers'],{'claude':1})

    def test_manual_confirmations_cannot_be_diagnosed_as_model_output(self):
        w,l,p=fixture()
        p['events'][0]['confirmed']=True
        with self.assertRaisesRegex(ValueError,'Manual'):
            diagnose_window(w,l,p,'nudity',False)

    def test_duplicate_categories_or_windows_cannot_double_count(self):
        w,l,p=fixture();row=('dev',w,l,p)
        with self.assertRaisesRegex(ValueError,'Duplicate categories'):
            report([row],{},['nudity','nudity'])
        with self.assertRaisesRegex(ValueError,'Duplicate diagnostic windows'):
            report([row,row],{},['nudity'])

    def test_frozen_inputs_require_complete_identity_bound_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'labeling'/'dev';output=root/'paired'
            w,l,p=fixture();project=dict(windows=[w])
            write_json(folder/'project.json',project)
            write_json(folder/'labels/clip.json',l)
            (folder/'clips').mkdir();clip=folder/'clips/clip.mp4';clip.write_bytes(b'original')
            contract=dict(project_identity=digest(project), expected_windows=['clip'],
                          sources={'clip':file_hash(clip)})
            contracts={'dev':contract};manifest=dict(project_contracts=contracts,identity=digest(contracts),complete=True)
            write_json(output/'run.json',manifest)
            p['run_identity']=manifest['identity'];write_json(output/'dev/clip.json',p)
            rows,_=load_frozen(root,['dev'],output)
            self.assertEqual(len(rows),1)
            for change in ('incomplete','identity','missing'):
                altered=copy.deepcopy(manifest);changed=copy.deepcopy(p)
                if change=='incomplete':altered['complete']=False
                if change=='identity':changed['run_identity']='stale'
                if change=='missing':changed.pop('trace')
                write_json(output/'run.json',altered);write_json(output/'dev/clip.json',changed)
                with self.assertRaises((ValueError,KeyError)):
                    rows,_=load_frozen(root,['dev'],output)
                    report(rows,{},['nudity'])
            write_json(output/'run.json',manifest);write_json(output/'dev/clip.json',p)
            clip.write_bytes(b'changed-source')
            with self.assertRaisesRegex(ValueError,'Source changed'):
                load_frozen(root,['dev'],output)

    def test_changed_projects_and_incomplete_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'labeling/dev';output=root/'paired'
            w,l,p=fixture();project=dict(windows=[w])
            write_json(folder/'project.json',project)
            write_json(folder/'labels/clip.json',l)
            contract=dict(project_identity=digest(project),expected_windows=['clip'],sources={})
            contracts={'dev':contract};manifest=dict(project_contracts=contracts,identity=digest(contracts),complete=True)
            write_json(output/'run.json',manifest)
            altered=copy.deepcopy(project);altered['windows'][0]['target']=[1,9]
            write_json(folder/'project.json',altered)
            with self.assertRaisesRegex(ValueError,'Project changed'):
                load_frozen(root,['dev'],output)
            write_json(folder/'project.json',project);l['categories'].pop('gore')
            write_json(folder/'labels/clip.json',l)
            with self.assertRaisesRegex(ValueError,'Incomplete labels'):
                load_frozen(root,['dev'],output)


if __name__=='__main__':
    unittest.main()
