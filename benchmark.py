"""Reproducible incremental-engine timing. This is not an accuracy evaluation."""
import argparse
import json
from pathlib import Path
import threading
import time

from analysis_engine import Models, ScanConfig, CHUNK, media_info


def benchmark(path, device, fps):
    import psutil
    process = psutil.Process()
    peak = [0]
    stopped = threading.Event()
    def measure():
        while not stopped.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=measure, daemon=True)
    monitor.start()
    config = ScanConfig(device=device, fps=fps, phrases=['banana'])
    models = Models()
    info = media_info(path)
    timings, detections = [], []
    started = time.perf_counter()
    try:
        for index in range(int((info['duration'] + CHUNK - 1e-5) // CHUNK)):
            before = time.perf_counter()
            found = models.analyze(path, index * CHUNK, min((index + 1) * CHUNK, info['duration']),
                                   info['duration'], config, info['has_audio'])
            timings.append(time.perf_counter() - before)
            detections.extend(found)
            print(f'Chunk {index}: {timings[-1]:.3f}s', flush=True)
    finally:
        stopped.set(); monitor.join()
    elapsed = time.perf_counter() - started
    import numpy as np
    return dict(media_seconds=info['duration'], wall_seconds=elapsed, realtime_factor=info['duration']/elapsed,
                cold_first_chunk_seconds=timings[0], warm_p95_chunk_seconds=float(np.percentile(timings[1:] or timings,95)),
                chunk_seconds=timings, peak_process_rss_mib=peak[0]/1024**2, backends=models.backends,
                fps=fps, analysis_config=config.model_dump(), accuracy_evaluation=False,
                speech_candidate_count=sum(e['category']=='language' for e in detections))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--fps',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=benchmark(args.media,args.device,args.fps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
