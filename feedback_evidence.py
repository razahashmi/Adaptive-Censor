"""Bounded local evidence retention and human-only dataset ingress for playback reports."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading
from uuid import UUID

import av
from fastapi import HTTPException

from core import run_ffmpeg
from labeling import (DEFINITIONS_REVISION, LABEL_CATEGORIES, is_complete, media_info,
                      read_json, validate_labels, write_json)
from provenance import digest, file_hash
from select_windows import normalize_title

REVIEW_PROJECT = 'feedback-review-v1'
TRAIN_PROJECT = 'training-feedback-human-v1'


def source_stamp(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def encode_clip(source, start, end, destination):
    # One small CPU encode; native frames, no model/GPU work or full-source hashing.
    run_ffmpeg(['-y', '-filter_threads', '1', '-ss', f'{start:.6f}', '-threads', '2', '-i', str(source), '-t', f'{end-start:.6f}',
                '-map', '0:v:0', '-map', '0:a:0?', '-vf', 'scale=w=min(1280\\,iw):h=-2,setsar=1',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '20', '-bf', '0',
                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ac', '2', '-b:a', '128k',
                '-sn', '-dn', '-map_metadata', '-1', '-map_chapters', '-1', '-movflags', '+faststart', str(destination)], timeout=90)
    info = media_info(destination)
    with av.open(str(destination)) as container:
        rate = container.streams.video[0].average_rate
    return dict(duration=info['duration'], has_audio=info['has_audio'], fps=float(rate) if rate else 24.)


class Evidence:
    def __init__(self, root, resolve_source, *, encoder=encode_clip, budget=512*1024**2):
        self.root = Path(root)
        self.folder = self.root/'feedback'/'evidence'
        self.resolve_source = resolve_source
        self.encoder = encoder
        self.budget = budget
        self.lock = threading.RLock()
        self.jobs = {}
        self.sessions = {}
        self.errors = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='feedback-clip')

    def directory(self, report_id):
        return self.folder/str(UUID(str(report_id)))

    def status(self, report_id):
        with self.lock:
            folder = self.directory(report_id)
            item = read_json(folder/'evidence.json', dict(status='none'))
            running = self.jobs.get(str(report_id))
            if item['status'] in ('queued', 'saving') and not (running and not running.done()):
                item = dict(status='interrupted', message='Clip saving was interrupted. Retry while the source is available.')
            if item['status']=='ready':
                try:
                    if source_stamp(folder/'clip.mp4') != item['clip_stamp']:
                        item = dict(status='changed', message='The saved clip changed or is missing. Keep it again before labeling.')
                except OSError:
                    item = dict(status='missing', message='The saved clip is missing. Keep it again while the source is available.')
            if str(report_id) in self.errors:
                item = dict(status='error', message=self.errors[str(report_id)])
            task = read_json(folder/'task.json', {})
            # Internal paths/source references are deliberately excluded from API responses.
            result = {k:item[k] for k in ('status', 'message', 'duration', 'bytes', 'sha256', 'offset') if k in item}
            if task:
                result['task'] = {k:task[k] for k in ('project', 'window', 'url', 'title', 'target')}
                labels = read_json(self.root/'labeling'/task['project']/'labels'/f"{task['window']}.json", {})
                result['task']['complete'] = is_complete(labels)
                result['task']['promoted'] = (self.root/'labeling'/TRAIN_PROJECT/'labels'/f"{task['window']}.json").is_file()
            return result

    def retain(self, report):
        report_id = report['id']
        with self.lock:
            status = self.status(report_id)
            if status['status'] in ('ready', 'queued', 'saving'):
                return status
            if report['status']=='withdrawn':
                raise HTTPException(409, 'This report was undone. Restore it in Saved reports before keeping a clip.')
            if sum(not future.done() for future in self.jobs.values())>=8:
                raise HTTPException(409, 'Clip queue is full. Retry after the current clips finish.')
            if self.used_bytes()>=self.budget:
                raise HTTPException(409, 'The local clip storage limit is reached. Remove retained clips before retrying.')
            source = Path(self.resolve_source(report))
            expected = {k:report['context']['source_ref'][k] for k in ('size', 'mtime_ns')}
            if source_stamp(source)!=expected:
                raise HTTPException(409, 'The source video changed after this report.')
            folder = self.directory(report_id)
            write_json(folder/'evidence.json', dict(status='queued'))
            self.errors.pop(report_id, None)
            self.jobs[report_id] = self.pool.submit(self.capture, deepcopy(report), source, expected)
            self.sessions[report_id] = report.get('session_id')
            # Release completed Future objects instead of retaining every result forever.
            self.jobs = {key:value for key,value in self.jobs.items() if key==report_id or not value.done()}
            self.sessions = {key:value for key,value in self.sessions.items() if key in self.jobs}
            return self.status(report_id)

    def busy_session(self, session_id):
        with self.lock:
            return any(self.sessions.get(key)==session_id and not future.done() for key,future in self.jobs.items())

    def used_bytes(self):
        return sum(path.stat().st_size for path in self.folder.glob('*/clip.mp4') if path.is_file())

    def failed(self, report_id, message):
        with self.lock:
            self.errors[str(report_id)] = message
            try:write_json(self.directory(report_id)/'evidence.json', dict(status='error', message=message))
            except OSError:pass

    def capture(self, report, source, expected):
        folder = self.directory(report['id'])
        temporary = folder/'clip.part.mp4'
        try:
            with self.lock:write_json(folder/'evidence.json', dict(status='saving'))
            if source_stamp(source)!=expected:
                raise ValueError('Source changed')
            start, end = report['window']
            if not 0 < end-start <= 12.001:
                raise ValueError('Invalid report window')
            info = self.encoder(source, start, end, temporary)
            if not .1 <= info['duration'] <= end-start+.3:
                raise ValueError('Unexpected retained duration')
            if source_stamp(source)!=expected:
                raise ValueError('Source changed during capture')
            identity = file_hash(temporary)
            with self.lock:
                old_size = (folder/'clip.mp4').stat().st_size if (folder/'clip.mp4').is_file() else 0
                if self.used_bytes()-old_size+temporary.stat().st_size>self.budget:
                    raise ValueError('Clip quota exceeded')
                temporary.replace(folder/'clip.mp4')
                write_json(folder/'evidence.json', dict(status='ready', **info, offset=start,
                    bytes=(folder/'clip.mp4').stat().st_size, sha256=identity,
                    clip_stamp=source_stamp(folder/'clip.mp4'), source_ref=report['context']['source_ref']))
        except Exception:
            message='Could not keep the clip. Check free storage and source availability, then retry.'
            with self.lock:
                self.errors[report['id']] = message
                try:write_json(folder/'evidence.json', dict(status='error', message=message))
                except OSError:pass
        finally:
            temporary.unlink(missing_ok=True)

    def verified_clip(self, report_id):
        with self.lock:
            folder = self.directory(report_id)
            info = read_json(folder/'evidence.json', {})
            if self.status(report_id)['status']!='ready' or file_hash(folder/'clip.mp4')!=info.get('sha256'):
                raise HTTPException(409, 'A complete, unchanged saved clip is required.')
            return folder/'clip.mp4', info

    def remove(self, report_id):
        with self.lock:
            if self.status(report_id)['status'] in ('saving', 'queued'):
                raise HTTPException(409, 'Wait for the clip to finish saving before removing it.')
            folder = self.directory(report_id)
            for name in ('clip.mp4', 'clip.part.mp4'):
                (folder/name).unlink(missing_ok=True)
            write_json(folder/'evidence.json', dict(status='removed'))
            self.errors.pop(str(report_id), None)
            return self.status(report_id)  # A copied labeling task remains independent.

    def make_task(self, report, title, start, end):
        with self.lock:
            source, info = self.verified_clip(report['id'])
            if report['status']!='confirmed':
                raise HTTPException(409, 'Confirm the issue before preparing precise labels.')
            if report['category'] not in LABEL_CATEGORIES:
                raise HTTPException(422, 'Choose a scene category first. Language/custom reports stay in the report queue.')
            title = normalize_title(title.strip())
            if len(title)<2 or not 0 <= start < end <= info['duration']+.001 or end-start<.1:
                raise HTTPException(422, 'Use the actual series/movie title and valid clip-relative review bounds.')
            end = min(end, info['duration'])
            identity = digest(dict(title=title, target=[start,end], sha256=info['sha256']))
            folder = self.directory(report['id'])
            existing = read_json(folder/'task.json', {})
            if existing:
                if existing['identity']!=identity:
                    raise HTTPException(409, 'This report already has a labeling task. Continue that task instead of replacing it.')
                return self.status(report['id'])
            window_id = 'feedback-'+UUID(report['id']).hex
            window = dict(id=window_id, title=title, episode=report['context']['title'], split='dev',
                stratum='playback-report', source=str(source), start=info['offset'], length=info['duration'],
                clip_offset=info['offset'], clip_duration=info['duration'], fps=info['fps'], target=[start,end],
                feedback_report=report['id'], clip_sha256=info['sha256'], model_exposed=True,
                evaluation_eligible=False, feedback_kind=report['kind'])
            self.write_window(REVIEW_PROJECT, window, source)
            task = dict(project=REVIEW_PROJECT, window=window_id, title=title, target=[start,end],
                        identity=identity, url=f'/label.html#{REVIEW_PROJECT}/{window_id}')
            write_json(folder/'task.json', task)
            return self.status(report['id'])

    def write_window(self, name, window, source, labels=None):
        folder = self.root/'labeling'/name
        project = read_json(folder/'project.json', dict(version=1, name=name, context=0,
                            definitions_revision=DEFINITIONS_REVISION, windows=[]))
        existing = next((w for w in project['windows'] if w['id']==window['id']), None)
        if existing and existing!=window:
            raise HTTPException(409, 'Existing dataset entry differs. It will not be overwritten.')
        (folder/'clips').mkdir(parents=True, exist_ok=True)
        destination = folder/'clips'/f"{window['id']}.mp4"
        if destination.is_file() and file_hash(destination)!=window['clip_sha256']:
            raise HTTPException(409, 'The existing dataset clip changed. It will not be overwritten.')
        if not destination.is_file():
            temporary = destination.with_suffix('.part.mp4')
            try:
                shutil.copyfile(source, temporary)
                if file_hash(temporary)!=window['clip_sha256']:raise HTTPException(409, 'Clip changed during copying')
                temporary.replace(destination)
            finally:temporary.unlink(missing_ok=True)
        if not existing:
            project['windows'].append(window)
            write_json(folder/'project.json', project)
        if labels is not None:
            path = folder/'labels'/f"{window['id']}.json"
            previous = read_json(path, {})
            if previous and previous!=labels:
                raise HTTPException(409, 'Training labels already exist. Review them explicitly before replacing them.')
            if not previous:write_json(path, labels)

    def promote(self, report):
        with self.lock:
            if report['status']!='confirmed':raise HTTPException(409, 'Confirm the issue first.')
            task = read_json(self.directory(report['id'])/'task.json', {})
            if not task:raise HTTPException(409, 'Prepare and complete a precise labeling task first.')
            folder = self.root/'labeling'/REVIEW_PROJECT
            project = read_json(folder/'project.json')
            window = next(w for w in project['windows'] if w['id']==task['window'])
            if digest(dict(title=normalize_title(window['title']), target=window['target'], sha256=window['clip_sha256']))!=task['identity']:
                raise HTTPException(409, 'The review task identity changed. It cannot enter training.')
            source = folder/'clips'/f"{window['id']}.mp4"
            if file_hash(source)!=window['clip_sha256']:raise HTTPException(409, 'The labeled clip changed.')
            labels = read_json(folder/'labels'/f"{window['id']}.json", {})
            if not is_complete(labels) or labels.get('labeler')!='human':
                raise HTTPException(409, 'A person must finish all scene categories in the label editor first.')
            if project.get('definitions_revision')!=DEFINITIONS_REVISION or labels.get('definitions_revision')!=DEFINITIONS_REVISION:
                raise HTTPException(409, 'These labels use older category definitions. Review them under the current definitions first.')
            clean = validate_labels(window, labels)
            clean['updated'] = labels.get('updated', clean['updated'])
            if not any(v['status']=='reviewed' for v in clean['categories'].values()):
                raise HTTPException(409, 'The clip has no reviewed categories to train on.')
            library = read_json(self.root/'labeling'/'library.json', {})
            title = normalize_title(window['title'])
            assignments = {}
            for key,value in library.get('splits', {}).items():
                assignments.setdefault(normalize_title(key), set()).add(value)
            if assignments.get(title)!={'train'}:
                raise HTTPException(409, 'This title is not permanently assigned to training. Review data stays separate.')
            source_title = normalize_title(report['context']['title'])
            if assignments.get(source_title,set()) & {'val','test'}:
                raise HTTPException(409, 'The source title belongs to validation or test and cannot enter training.')
            for path in (self.root/'labeling').glob('*/project.json'):
                for other in read_json(path, {}).get('windows', []):
                    if normalize_title(other.get('title',''))==title and other.get('split') in ('val','test'):
                        raise HTTPException(409, 'This title already appears in a held-out project and cannot enter training.')
            clean['drafts_shown'] = True  # Report review has exposed model decisions; never claim blind labels.
            promoted = dict(window, split='train', source=str(source), feedback_review_identity=task['identity'],
                            feedback_labels_identity=digest(clean))
            self.write_window(TRAIN_PROJECT, promoted, source, clean)
            return dict(project=TRAIN_PROJECT, window=window['id'], url=f"/label.html#{TRAIN_PROJECT}/{window['id']}",
                        trained=False, model_changed=False, evaluation_eligible=False)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
