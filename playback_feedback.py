"""Local, unverified playback reports. Never an input to training or evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core import CATEGORIES, compile_plan

Category = Literal['unspecified', 'nudity', 'sexual_activity', 'violence', 'gore', 'kissing', 'language', 'custom']
Status = Literal['pending', 'confirmed', 'dismissed', 'withdrawn']


class FeedbackRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def validated(request):
            try:
                return await handler(request)
            except RequestValidationError as error:
                # Raw JSON NaN/Infinity in FastAPI's error input cannot itself be JSON encoded.
                raise HTTPException(422, [{k:e[k] for k in ('loc', 'msg', 'type')} for e in error.errors()]) from error
        return validated


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Rule(StrictModel):
    visual: Literal['allow', 'blur', 'hide']
    audio: Literal['allow', 'beep', 'mute']
    blur: float = Field(ge=1, le=100)
    min_confidence: float = Field(ge=0, le=1)
    min_severity: float = Field(default=1, ge=1, le=3)
    scope: Literal['region', 'frame'] = 'frame'


class Policy(StrictModel):
    pre_padding: float = Field(ge=0, le=1)
    post_padding: float = Field(ge=0, le=1)
    categories: dict[str, Rule] = Field(max_length=7)


class Action(StrictModel):
    visual: Literal['allow', 'blur', 'hide']
    audio: Literal['allow', 'beep', 'mute']
    blur: float = Field(ge=0, le=100)
    regional: bool = False


class Report(StrictModel):
    id: UUID
    session_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    kind: Literal['wrongly_censored', 'missed_content']
    position: float = Field(ge=0)
    generation: int = Field(ge=0)
    snapshot_updated: int = Field(ge=0)
    policy: Policy
    viewer_action: Action
    retain_clip: bool = False


class LabelTask(StrictModel):
    title: str = Field(min_length=2, max_length=200)
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class Details(StrictModel):
    revision: int = Field(ge=1)
    category: Category | None = None
    note: str | None = Field(default=None, max_length=1000)
    status: Status | None = None


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root):
        self.path = Path(root) / 'feedback' / 'reports.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, '
                       'created TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, report_id):
        with self.connect() as db:
            row = db.execute('SELECT body FROM reports WHERE id=?', (str(report_id),)).fetchone()
        if row is None:
            raise HTTPException(404, 'Report not found')
        return json.loads(row[0])

    def existing(self, report_id, fingerprint):
        with self.connect() as db:
            row = db.execute('SELECT fingerprint, body FROM reports WHERE id=?', (report_id,)).fetchone()
        if row:
            if row[0] != fingerprint:
                raise HTTPException(409, 'This report ID was already used for a different moment')
            return json.loads(row[1])

    def add(self, request, context):
        payload = request.model_dump(mode='json')
        fingerprint_payload = dict(payload)
        if not request.retain_clip:
            fingerprint_payload.pop('retain_clip')  # Keep pre-retention pending requests idempotent after upgrades.
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
        report_id = str(request.id)
        existing = self.existing(report_id, fingerprint)
        if existing:
            return existing  # A retry still succeeds after the source session has been removed.
        if set(request.policy.categories) != set(CATEGORIES):
            raise HTTPException(422, 'Include the rules for every category')
        captured = context(request.session_id)
        state = captured.pop('snapshot')
        if request.position > state['duration']:
            raise HTTPException(422, 'Playback time is outside this media')
        start, end = max(0, request.position - 10), min(state['duration'], request.position + 2)
        nearby = [e for e in state['events'] if e['start'] < end and e['end'] > start]
        policy = request.policy.model_dump()
        plan = compile_plan(nearby, policy, state['duration'])
        changed = request.generation != state['generation'] or request.snapshot_updated != state['updated']
        created = now()
        record = dict(schema_version=1, id=report_id, created_at=created, updated_at=created, revision=1,
                      status='pending', category='unspecified', note='', verified_label=False,
                      **{k:v for k,v in payload.items() if k != 'id'},
                      window=[start, end], context={**captured, 'title':state['title'], 'duration':state['duration'],
                      'live':state['live'], 'retained_start':state['retained_start'], 'config':state['config'],
                      'generation':state['generation'], 'snapshot_updated':state['updated'],
                      'analysis_status':state['status'], 'context_changed_since_playback':changed,
                      'analyzed_at_receipt':any(a <= request.position < b for a,b in state['coverage']),
                      'events_at_receipt':nearby[:500], 'events_truncated':len(nearby)>500,
                      'plan_at_receipt':[e for e in plan if e['start'] < end and e['end'] > start][:500]})
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO reports VALUES (?,?,?,?,?,?)',
                       (report_id, fingerprint, created, 'pending', 1, json.dumps(record)))
        return self.existing(report_id, fingerprint)

    def update(self, report_id, details):
        record = self.get(report_id)
        record.update(details.model_dump(exclude_none=True, exclude={'revision'}))
        record.update(revision=details.revision + 1, updated_at=now())
        with self.connect() as db:
            result = db.execute('UPDATE reports SET status=?,revision=?,body=? WHERE id=? AND revision=?',
                                (record['status'], record['revision'], json.dumps(record), str(report_id), details.revision))
            if not result.rowcount:
                raise HTTPException(409, 'Report changed in another tab. Reload it before editing.')
        return record

    def rows(self, status, limit, offset):
        where = 'WHERE status=?' if status != 'all' else ''
        args = (status,) if where else ()
        with self.connect() as db:
            total = db.execute(f'SELECT COUNT(*) FROM reports {where}', args).fetchone()[0]
            rows = db.execute(f'SELECT body FROM reports {where} ORDER BY created DESC,id LIMIT ? OFFSET ?',
                              (*args, limit, offset)).fetchall()
        return total, [json.loads(row[0]) for row in rows]

    def export(self):
        # Stream a consistent snapshot; exports exclude withdrawn reports.
        with self.connect() as db:
            for (body,) in db.execute("SELECT body FROM reports WHERE status != 'withdrawn' ORDER BY created,id"):
                yield body + '\n'


def router(root, context, available, evidence=None):
    store = Store(root)
    api = APIRouter(prefix='/api/feedback', route_class=FeedbackRoute)

    def decorate(row):
        return dict(row, media_available=available(row),
                    evidence=evidence.status(row['id']) if evidence else dict(status='none'))

    @api.post('')
    def report(request: Report):
        row = store.add(request, context)
        if evidence and row.get('retain_clip') and row['status']!='withdrawn' and evidence.status(row['id'])['status']!='removed':
            try:evidence.retain(row)
            except (HTTPException, OSError) as error:
                evidence.failed(row['id'], error.detail if isinstance(error, HTTPException) else 'Clip storage is unavailable.')
        return decorate(row)

    @api.get('')
    def reports(status: Literal['all', 'pending', 'confirmed', 'dismissed', 'withdrawn'] = 'pending',
                limit: int = Query(default=50, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        total, rows = store.rows(status, limit, offset)
        return dict(total=total, reports=[decorate(row) for row in rows])

    @api.get('/export')
    def export():
        def records():
            for value in store.export():
                row=json.loads(value)
                if evidence:row['evidence']=evidence.status(row['id'])
                yield json.dumps(row)+'\n'
        return StreamingResponse(records(), media_type='application/x-ndjson',
                                 headers={'Content-Disposition':'attachment; filename="playback-reports.jsonl"'})

    @api.get('/{report_id}')
    def get(report_id: UUID):
        return decorate(store.get(report_id))

    @api.patch('/{report_id}')
    def update(report_id: UUID, details: Details):
        return decorate(store.update(report_id, details))

    def keeper():
        if evidence is None:raise HTTPException(503, 'Clip retention is not configured on this server.')
        return evidence

    @api.post('/{report_id}/evidence')
    def keep_evidence(report_id: UUID):
        row=store.get(report_id)
        keeper().retain(row)
        return decorate(row)

    @api.get('/{report_id}/evidence')
    def evidence_clip(report_id: UUID):
        store.get(report_id)
        path,_=keeper().verified_clip(report_id)
        return FileResponse(path, media_type='video/mp4')

    @api.delete('/{report_id}/evidence')
    def remove_evidence(report_id: UUID):
        row=store.get(report_id)
        keeper().remove(report_id)
        return decorate(row)

    @api.post('/{report_id}/labeling')
    def label_report(report_id: UUID, task: LabelTask):
        row=store.get(report_id)
        keeper().make_task(row, task.title, task.start, task.end)
        return decorate(row)

    @api.post('/{report_id}/training')
    def promote_report(report_id: UUID):
        return keeper().promote(store.get(report_id))

    return api
