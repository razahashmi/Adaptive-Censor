"""Build a short showcase with real licensed footage and an audible example.

Reviewed timings use the actual project renderer; no recognition model is run.
Source footage and intermediate media stay outside the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from core import compile_plan, default_policy, render_video, run_ffmpeg
from build_source_release import REVIEWED_MEDIA_SOURCES

WIDTH, HEIGHT, FPS, RATE = 1280, 720, 24, 48000
BG = (10, 17, 25); SURFACE = (19, 29, 40); INK = (238, 244, 246)
MUTED = (157, 177, 188); MINT = (111, 237, 193)
FILES = ('adaptive-censor-demo.mp4', 'adaptive-censor-preview.gif', 'adaptive-censor-poster.png')
KISS_SPAN = (4.05, 5.65)
# Actual profanity: 'dick', 360.26–360.54s. Reviewed margins cover the whole word.
SPEECH_SOURCE_START = 357.8
SPEECH_SPAN = (2.36, 2.90)
SCENES = [('intro', 3), ('warning', 3), ('kissing', 6), ('collage', 6), ('options', 3),
          ('ui-controls', 8),
          ('sound-original', 3), ('sound-beep', 3), ('sound-mute', 3),
          ('ui-review', 8), ('improve', 9), ('credits', 8)]
DURATION = sum(seconds for _, seconds in SCENES)
REPOSITORY = 'github.com/razahashmi/Adaptive-Censor'


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def font(size, bold=False):
    for path in [Path('C:/Windows/Fonts')/('segoeuib.ttf' if bold else 'segoeui.ttf'),
                 Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]:
        if path.exists(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


FONTS = {key: font(size, bold) for key, size, bold in
         [('title', 74, True), ('heading', 44, True), ('body', 25, False),
          ('sub', 22, False), ('small', 17, False), ('label', 16, True), ('large', 31, True)]}


def text(frame, xy, value, kind='body', color=INK):
    ImageDraw.Draw(frame).text(xy, value, font=FONTS[kind], fill=color)


def base(source=None):
    frame = Image.new('RGB', (WIDTH, HEIGHT), BG); d = ImageDraw.Draw(frame)
    d.rounded_rectangle((48, 34, 81, 67), radius=8, fill=MINT)
    d.rectangle((54, 45, 75, 55), fill=BG)
    text(frame, (94, 35), 'ADAPTIVE CENSOR', 'label')
    d.rounded_rectangle((1006, 32, 1232, 69), radius=18, fill=SURFACE)
    text(frame, (1026, 40), 'EXPERIMENTAL PREVIEW', 'label', MINT)
    # Source attribution appears once on the end card; the footer stays empty.
    return frame


def chip(frame, box, label, accent=False):
    ImageDraw.Draw(frame).rounded_rectangle(box, radius=10, fill=MINT if accent else SURFACE)
    text(frame, (box[0]+16, box[1]+10), label, 'label', BG if accent else INK)


def fit(pixels, size):
    return ImageOps.pad(Image.fromarray(pixels), size, method=Image.Resampling.LANCZOS, color=BG)


def frames(path):
    with av.open(path) as container:
        for frame in container.decode(video=0): yield frame.to_ndarray(format='rgb24')


def render_filter(source, target, duration, category, *, visual='allow', audio='allow',
                  strength=100, region=None, span=None):
    policy = default_policy(); policy.update(pre_padding=0, post_padding=0)
    policy['categories'][category].update(visual=visual, audio=audio, blur=strength,
                                         scope='region' if region else 'frame')
    start, end = span or (0, duration)
    event = dict(start=start, end=end, category=category, confidence=1, severity=2,
                 enabled=True, source='manual', note='Reviewed showcase interval, not an automatic prediction')
    if region: event['region'] = region
    plan = compile_plan([event], policy, duration)
    render_video(source, target, plan)
    return dict(events=[event], policy=policy, compiled_plan=plan)


def kiss_frame(raw, changed, second):
    frame = base('kissing'); d = ImageDraw.Draw(frame)
    text(frame, (48, 101), 'Keep the story. Choose the filter.', 'heading')
    text(frame, (48, 173), 'ORIGINAL · ADULT KISSING', 'label', MUTED)
    text(frame, (652, 173), 'FILTERED · STRONG BLUR', 'label', MINT)
    for x, pixels in [(48, raw), (652, changed)]: frame.paste(fit(pixels, (580, 326)), (x, 203))
    chip(frame, (48, 566, 432, 608), 'KISSING · BLUR STRENGTH 100 / 100', True)
    left, width = 465, 767
    d.rounded_rectangle((left, 581, left+width, 591), radius=4, fill=SURFACE)
    d.rectangle((left+width*KISS_SPAN[0]/6, 581, left+width*KISS_SPAN[1]/6, 591), fill=(61, 116, 101))
    x = left+width*second/6; d.rectangle((x-2, 575, x+2, 597), fill=MINT)
    text(frame, (48, 640), 'Manually reviewed interval · actual renderer output · detection remains experimental', 'small', MUTED)
    return frame


def collage_frame(pictures):
    frame = base('film')
    text(frame, (48, 101), 'Four filters. Your choice.', 'heading')
    for (x, y), (label, pixels) in zip([(48, 190), (652, 190), (48, 438), (652, 438)],
                                     zip(['LIGHT BLUR · 20', 'STRONG BLUR · 100', 'SELECTED REGION · STRONG BLUR', 'COMPLETE CONCEALMENT'], pictures)):
        text(frame, (x, y-26), label, 'label', MINT)
        frame.paste(fit(pixels, (580, 200)), (x, y))
        if pixels.max() < 4: text(frame, (x+172, y+84), 'Scene concealed', 'sub', MUTED)
    text(frame, (48, 650), 'Fictional brain prop · reviewed gore interval and region · original scene not repeated', 'small', MUTED)
    return frame


def sound_frame(pixels, mode, second):
    frame = base('film'); d = ImageDraw.Draw(frame)
    text(frame, (48, 101), 'Your word list. Your sound rules.', 'heading')
    order = {'original': 1, 'beep': 2, 'mute': 3}[mode]
    chip(frame, (48, 167, 432, 210), f'{order} / 3 · '+{'original':'ORIGINAL AUDIO', 'beep':'SELECTED WORD BEEPED', 'mute':'SELECTED WORD MUTED'}[mode], True)
    text(frame, (674, 178), 'LISTEN · '+('original contains profanity' if mode=='original' else 'abusive word filtered'), 'sub', MUTED)
    frame.paste(fit(pixels, (960, 400)), (160, 231))
    # The overlay is a caption, not a fabricated app screen or ASR recording.
    d.rectangle((160, 527, 1120, 631), fill=SURFACE)
    word = {'original':'dick', 'beep':'[BEEP]', 'mute':'[MUTED]'}[mode]
    text(frame, (185, 542), '“I was young,', 'body')
    text(frame, (185, 578), f'and a {word}.”', 'body')
    text(frame, (48, 650), 'Real film dialogue · reviewed word timing · mute/beep replaces the full mix during that interval', 'small', MUTED)
    return frame


def options_frame():
    """An editorial control overview, with labels taken from the working player."""
    frame = base('options'); d = ImageDraw.Draw(frame)
    text(frame, (48, 101), 'Set your boundaries.', 'heading')
    text(frame, (48, 158), 'Choose picture and sound independently for each category.', 'sub', MUTED)
    rows = [('ANATOMY / NUDITY','Blur · detected regions','Keep sound',70),
            ('SEXUAL ACTIVITY','Conceal completely','Mute',100),
            ('KISSING','Blur','Keep sound',40),
            ('VIOLENCE','Blur','Keep sound',35),
            ('GORE','Conceal completely','Mute',100),
            ('LANGUAGE','Keep picture','Beep',None)]
    for index,(name,picture,sound,strength) in enumerate(rows):
        x = 48+(index%3)*404; y = 221+(index//3)*204
        d.rounded_rectangle((x,y,x+376,y+184),radius=14,fill=SURFACE,outline=(43,60,73),width=1)
        text(frame,(x+18,y+15),name,'label',MINT)
        text(frame,(x+18,y+52),'Picture: '+picture,'small')
        text(frame,(x+18,y+84),'Sound: '+sound,'small')
        if strength is not None:
            text(frame,(x+18,y+119),'Blur strength','small',MUTED)
            text(frame,(x+318,y+119),str(strength),'small')
            d.rounded_rectangle((x+18,y+155,x+347,y+162),radius=3,fill=BG)
            d.rounded_rectangle((x+18,y+155,x+18+329*strength/100,y+162),radius=3,fill=MINT)
        else:
            text(frame,(x+18,y+121),'Your word list + exceptions','small',MUTED)
            text(frame,(x+18,y+150),'Mute or beep matched intervals','small',MUTED)
    text(frame,(48,649),'Controls overview · illustrative settings · blur strength applies when Blur is selected','small',MUTED)
    return frame


def poster_frame():
    """Original promotional artwork; no third-party source imagery."""
    frame = base('options'); d = ImageDraw.Draw(frame)
    text(frame, (48, 169), 'Your media.', 'title')
    text(frame, (48, 266), 'Your rules.', 'title', MINT)
    text(frame, (48, 396), 'Filter video + sound. Stay in control.', 'body')
    chip(frame, (48, 478, 330, 522), 'LOCAL · OPEN SOURCE', True)
    for (x,y), label, detail in zip([(754,180),(1000,180),(754,378),(1000,378)],
            ['BLUR','CONCEAL','MUTE','BEEP'],['Choose strength','Hide the picture','Quiet an interval','Replace a word']):
        d.rounded_rectangle((x,y,x+230,y+170),radius=16,fill=SURFACE)
        text(frame,(x+18,y+45),label,'large',MINT)
        text(frame,(x+18,y+108),detail,'small')
    text(frame,(48,620),'Watch the one-minute walkthrough →','large')
    return frame


def ui_frame(captures, name, second):
    """Edited crops from authentic UI screenshots, with explicit demo context."""
    frame=base('kissing')
    if name=='ui-controls':
        index=min(int(second/2),3)
        key=['controls-20','controls-40','controls-100','controls-hide'][index]
        im=Image.open(captures/(key+'.jpg')).convert('RGB')
        text(frame,(48,101),'Change the filter while you watch.','heading')
        text(frame,(48,166),'REAL PLAYER · REVIEWED TIMELINE','label',MINT)
        frame.paste(im.crop((51,525,858,861)).resize((807,336),Image.Resampling.LANCZOS),(48,211))
        frame.paste(im.crop((883,1370,1214,1530)),(895,255))
        text(frame,(895,438),['LIGHT · 20','MEDIUM · 40','STRONG · 100','CONCEALED'][index],'large',MINT)
        text(frame,(895,496),'Picture + sound','sub')
        text(frame,(895,532),'chosen separately.','sub')
        text(frame,(48,594),'A reviewed interval drives the effect; blur strength changes its appearance.','sub')
    else:
        stage=0 if second<2.5 else 1 if second<5 else 2
        key=['review-before','report-saved','review-fixed'][stage]
        im=Image.open(captures/(key+'.jpg')).convert('RGB')
        text(frame,(48,101),'A wrong filter? Report it. Correct it.','heading')
        text(frame,(48,166),'STAGED FALSE MATCH · ACTUAL REPORT AND CORRECTION','label',MINT)
        frame.paste(im.crop((51,525,858,861)),(48,207))
        text(frame,(895,254),['1 · WRONG FILTER','2 · REPORT SAVED','3 · EXCLUDE MATCH'][stage],'label',MINT)
        for i,line in enumerate([['A clean moment','was concealed.'],['One click saves','the local report.'],['Picture restored.','Review stays yours.']][stage]):
            text(frame,(895,303+i*40),line,'sub')
        if stage==0:
            # Actual feedback controls, before saving; stop above the metrics.
            crop=im.crop((51,861,858,966))
        elif stage==1:
            crop=im.crop((51,861,858,966))
        else:
            # Disabled interval and the real Restore button after Exclude.
            crop=im.crop((76,1996,832,2058))
        crop.thumbnail((807,100),Image.Resampling.LANCZOS)
        frame.paste(crop,(48,548))
        text(frame,(895,441),'Reports stay local.','sub')
        text(frame,(895,481),'No automatic training.','sub')
    text(frame,(48,650),'Edited authentic UI stills · licensed clip · manual demo events · no detector or speed claim','small',MUTED)
    return frame


def card(name):
    if name == 'options': return options_frame()
    frame = base()
    if name == 'intro':
        text(frame, (48, 185), 'Adaptive Censor', 'title')
        text(frame, (48, 305), 'Your media. Your rules.', 'large', MINT)
        text(frame, (48, 380), 'Local video + podcasts. Adjustable picture + sound.', 'body')
        chip(frame, (48, 478, 260, 521), 'OPEN SOURCE', True)
        return frame
    if name == 'warning':
        text(frame,(48,175),'Content warning','heading',MINT)
        for index,line in enumerate(['Adult kissing', 'Filtered fictional organ prop', 'Original audio contains profanity']):
            text(frame,(48,285+index*60),line,'body')
        text(frame,(48,530),'Original and filtered comparisons follow.','sub',MUTED)
        return frame
    if name == 'credits':
        text(frame,(48,165),'Try the preview. Help improve it.','heading')
        text(frame,(48,270),'SET UP · TEST · CONTRIBUTE','label',MINT)
        text(frame,(48,315),REPOSITORY,'large')
        chip(frame,(48,408,480,452),'LOCAL · OPEN SOURCE · MIT-LICENSED CODE',True)
        # Compact but readable attribution, shown only during the final 8 seconds.
        text(frame,(48,559),'Kissing: Anna Tarazevich / Pexels · Pexels License · video 5763324','small',MUTED)
        text(frame,(48,587),'Film + dialogue: Tears of Steel (2012) · (CC) Blender Foundation | mango.blender.org · CC BY 3.0','small',MUTED)
        text(frame,(48,625),'Full sources, terms and changes: docs/MEDIA_CREDITS.md','small',MUTED)
        return frame
    title, rows = {
        'workflow': ('Filtering that follows your choices.', [
            ('CHOOSE THE CATEGORY', 'Anatomy / nudity · sexual activity · kissing · violence · gore · language'),
            ('ANALYZE → REVIEW → PLAY', 'Local models suggest moments. You correct timing, regions and mistakes.'),
            ('ADJUST AS YOU WATCH', 'Picture and sound are independent. Rule changes need no source re-encode.')]),
        'home': ('A home server is the next step.', [
            ('TODAY', 'Analyze on one computer. Export a filtered MP4 for your other devices.'),
            ('PLANNED', 'Analyze once, then serve authenticated clients with household profiles.'),
            ('CURRENT LIMIT', 'Local-only service. Whole-home / DRM streaming filtering is not included.')]),
        'improve': ('Build better filtering together.', [
            ('RUN LOCALLY', 'Try the controls demo. Add models to analyze your own files.'),
            ('LABEL → EVALUATE → IMPROVE', 'Contribute reviewed examples, code and hardware tests. Measure misses + speed.'),
            ('HOME USE', 'Today: export filtered MP4s. Authenticated household streaming is planned.')])
    }[name]
    text(frame, (48, 109), title, 'heading')
    for index, (label, detail) in enumerate(rows):
        y = 229+index*122
        text(frame, (48, y), label, 'label', MINT)
        text(frame, (48, y+35), detail, 'sub')
    text(frame, (48, 640), 'Reviewed examples demonstrate controls, not detection accuracy. No creator or performer endorsement.', 'small', MUTED)
    return frame


def build(source, film, output, work, captures):
    for key, path in dict(kissing=source, film=film).items():
        if sha(path) != REVIEWED_MEDIA_SOURCES[key]['source_sha256']:
            raise ValueError('Use the exact licensed source/checksum in docs/MEDIA_CREDITS.md: '+key)
    output.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    capture_names=['controls-'+s for s in ('20','40','100','hide')]+['review-before','report-saved','review-fixed']
    capture_manifest=json.loads((captures/'CAPTURES.json').read_text(encoding='utf-8'))
    for name in capture_names:
        if sha(captures/(name+'.jpg')) != capture_manifest['files'][name+'.jpg']['sha256']:
            raise ValueError('UI capture differs from its reviewed manifest: '+name)
    poster_frame().save(output/FILES[2])
    def cut(path, name, start, length, sound=False):
        args = ['-y', '-ss', str(start), '-i', str(path), '-t', str(length), '-vf', 'fps=24,scale=1280:-2',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p']
        args += ['-af', 'loudnorm=I=-18:TP=-2:LRA=7', '-ac', '1', '-ar', str(RATE), '-c:a', 'aac', '-b:a', '192k'] if sound else ['-an']
        run_ffmpeg(args+['-map_metadata', '-1', str(work/name)])
    cut(source, 'kissing-source.mp4', 0, 6)
    cut(film, 'gore-source.mp4', 57, 4)
    cut(film, 'sound-original.mp4', SPEECH_SOURCE_START, 3, True)
    decisions = {'kissing': render_filter(work/'kissing-source.mp4', work/'kissing-filtered.mp4', 6, 'kissing', visual='blur', span=KISS_SPAN)}
    for name, visual, strength, region in [('light','blur',20,None), ('strong','blur',100,None),
                                         ('region','blur',100,[.21,.02,.57,.98]), ('hide','hide',100,None)]:
        decisions[name] = render_filter(work/'gore-source.mp4', work/(name+'.mp4'), 4, 'gore', visual=visual, strength=strength, region=region)
    for mode in ('beep', 'mute'):
        decisions[mode] = render_filter(work/'sound-original.mp4', work/('sound-'+mode+'.mp4'), 3, 'language', audio=mode, span=SPEECH_SPAN)
    (work/'reviewed-demo-timelines.json').write_text(json.dumps(decisions, indent=2), encoding='utf-8')
    # Decode audio from the real renderer outputs, preserving the audible comparison.
    audio = np.zeros(DURATION*RATE, dtype=np.float32)
    offsets = {}; elapsed = 0
    for name, seconds in SCENES:
        offsets[name] = elapsed; elapsed += seconds
    for mode in ('original','beep','mute'):
        offset = offsets['sound-'+mode]
        result = run_ffmpeg(['-i', str(work/('sound-'+mode+'.mp4')), '-vn', '-ac', '1', '-ar', str(RATE), '-f', 'f32le', 'pipe:1'])
        samples = np.frombuffer(result.stdout, dtype='<f4')[:3*RATE]
        audio[offset*RATE:offset*RATE+len(samples)] = samples
    audio_file = work/'showcase-audio.f32'; audio.tofile(audio_file)
    video_file = work/'showcase-picture.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-threads', '4', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', '-map_metadata', '-1', str(video_file)]
    with (work/'encode.log').open('wb') as log:
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for name, seconds in SCENES:
                if name == 'kissing':
                    for n,(raw,changed) in enumerate(zip(frames(work/'kissing-source.mp4'),frames(work/'kissing-filtered.mp4'))):
                        picture = kiss_frame(raw,changed,n/FPS)
                        encoder.stdin.write(picture.tobytes())
                elif name == 'collage':
                    # Four synchronized outputs, repeated only to fill the chosen chapter.
                    count = 0
                    for _ in range(2):
                        for pictures in zip(*(frames(work/(key+'.mp4')) for key in ('light','strong','region','hide'))):
                            if count >= seconds*FPS: break
                            encoder.stdin.write(collage_frame(pictures).tobytes())
                            count += 1
                elif name.startswith('ui-'):
                    for n in range(seconds*FPS):
                        encoder.stdin.write(ui_frame(captures,name,n/FPS).tobytes())
                elif name.startswith('sound-'):
                    for n,pixels in enumerate(frames(work/(name+'.mp4'))):
                        encoder.stdin.write(sound_frame(pixels,name.split('-')[1],n/FPS).tobytes())
                else:
                    picture = card(name)
                    pixels = picture.tobytes()
                    for _ in range(seconds*FPS): encoder.stdin.write(pixels)
                print('Rendered', name, flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=120): raise RuntimeError('Picture encoding failed; inspect local build log')
        except BaseException:
            encoder.kill();encoder.wait();raise
    run_ffmpeg(['-y','-i',str(video_file),'-f','f32le','-ar',str(RATE),'-ac','1','-i',str(audio_file),
                '-map','0:v:0','-map','1:a:0','-t',str(DURATION),'-c:v','copy','-c:a','aac','-b:a','192k','-map_metadata','-1',
                '-metadata','title=Adaptive Censor — one-minute introduction',
                '-metadata','comment=Anna Tarazevich / Pexels; Tears of Steel (CC) Blender Foundation; reviewed filters; see docs/MEDIA_CREDITS.md',
                '-movflags','+faststart',str(output/FILES[0])])
    run_ffmpeg(['-y','-ss',str(offsets['kissing']+4),'-t','2','-i',str(output/FILES[0]),'-filter_complex',
                'fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160[p];[b][p]paletteuse=dither=bayer:bayer_scale=3',
                '-loop','0','-an',str(output/FILES[1])])
    manifest = dict(schema_version=1,license='MIT AND LicenseRef-Pexels AND CC-BY-3.0',sources=REVIEWED_MEDIA_SOURCES,
        verified_date='2026-10-18',credits_file='docs/MEDIA_CREDITS.md',duration_seconds=DURATION,
        changes='Clean opening title with open-source branding; MIT-licensed code wording only on the end card; separate content warning before footage; one silent kissing excerpt; four-panel organ-prop filter collage; control illustration; edited authentic player screenshots; staged false-match reporting and exclusion; real profanity original/beeped/muted; captions and layouts. Compact source credits appear only on the final card; no repeated footer.',
        demonstration='Manually reviewed visual and speech intervals through the actual renderer. UI chapters use edited authentic stills on a manual timeline; the false match is deliberately staged. No automatic-recognition, speed or accuracy claim; no synthetic voice.',
        ui_capture_manifest_sha256=sha(captures/'CAPTURES.json'),
        reviewed_intervals=dict(kissing=KISS_SPAN,speech_in_excerpt=SPEECH_SPAN,collage=[0,4]),assets={})
    for name in FILES:
        ids=list(REVIEWED_MEDIA_SOURCES) if name.endswith('.mp4') else ['kissing'] if name.endswith('.gif') else []
        manifest['assets']['docs/media/'+name]=dict(sha256=sha(output/name),bytes=(output/name).stat().st_size,
            sources=ids,licenses=sorted({'MIT'}|{REVIEWED_MEDIA_SOURCES[key]['license'] for key in ids}))
    (output.parent.parent/'media-assets.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Complete:',DURATION,'seconds, picture + real speech,',round((output/FILES[0]).stat().st_size/1024**2,2),'MiB',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--film-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'docs/media')
    parser.add_argument('--work',type=Path,help='Optional inspection folder outside the public tree')
    parser.add_argument('--ui-capture',type=Path,required=True,help='Reviewed authentic screenshot pack; see docs/MEDIA_CREDITS.md')
    args = parser.parse_args()
    if args.work: build(args.source,args.film_source,args.output,args.work,args.ui_capture)
    else:
        with tempfile.TemporaryDirectory(prefix='adaptive-public-demo-') as temporary:
            build(args.source,args.film_source,args.output,Path(temporary),args.ui_capture)
