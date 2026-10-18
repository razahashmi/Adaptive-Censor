"""Confirm or reject every proposed learned interval, including head decisions.

Existing selective verification stays additive. This explicit experimental mode
never allows an unverified head decision to bypass a failed or uncertain check.
"""
from learned_runtime import GRID, VERIFIABLE, uncertain_spans


def confirm_outputs(entries, events, check, low, high, maximum_checks=12):
    spans=uncertain_spans(dict(trace=entries,events=[]),longest=.5)
    kept=[dict(e,review_required=True) for e in events if e['category'] not in VERIFIABLE]
    # Bound expensive checks across categories. Unanswered decisions require review.
    buckets={c:[] for c in VERIFIABLE}
    for c,a,b in spans:
        scores=[e['score'] for e in entries if e['category']==c and a<=e['sample_time']<=b]
        buckets[c].append((max(scores),c,a,b))
    for bucket in buckets.values():bucket.sort(reverse=True)
    chosen=[]
    while len(chosen)<maximum_checks and any(buckets.values()):
        for c in VERIFIABLE:
            if buckets[c] and len(chosen)<maximum_checks:chosen.append(buckets[c].pop(0)[1:])
    checked_ranges={c:[] for c in VERIFIABLE};trace=[]
    for category,first,last in sorted(chosen,key=lambda s:s[1]):
        middle=(first+last)/2
        if not low<=middle<high:continue
        evidence=check(category,middle)
        if evidence.get('state') not in ('present','absent','uncertain'):
            raise ValueError('Invalid learned confirmation state')
        start=max(low,first-GRID/2);end=min(high,last+GRID/2)
        checked_ranges[category].append((start,end))
        trace.append(dict(stage='learned_confirmation',category=category,start=start,end=end,
                          sample_time=middle,state=evidence['state'],detail=evidence['observation']))
        if evidence['state']=='absent':continue
        kept.append(dict(start=start,end=end,sample_time=middle,category=category,confidence=1,severity=2,
                         enabled=True,review_required=evidence['state']!='present',source='Confirmed learned interval',
                         evidence=evidence['label'],score_type='decision',note=evidence['observation']))
    for e in events:
        if e['category'] in VERIFIABLE and not any(a<=e['sample_time']<b for a,b in checked_ranges[e['category']]):
            kept.append(dict(e,review_required=True,note=e.get('note','')+' [Confirmation budget exhausted; review required]'))
    return trace,kept
