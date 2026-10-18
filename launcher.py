"""Start/stop this project's loopback server without keeping a console open."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
import httpx
import psutil
from model_setup import prepare_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'stop'])
    parser.add_argument('--port', type=int, default=8502)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = prepare_environment()
    record = root / 'server-process.json'
    if args.command == 'stop':
        if not record.exists():
            print('No server started by this launcher was recorded.'); return
        state = json.loads(record.read_text(encoding='utf-8'))
        try:
            process = psutil.Process(state['pid'])
            if abs(process.create_time() - state['created']) > .01 or str(Path(__file__).with_name('server.py')) not in process.cmdline():
                raise RuntimeError('The recorded PID no longer belongs to this server; no process was stopped.')
            for child in process.children(recursive=True):
                child.terminate()
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        record.unlink(missing_ok=True)
        print('Adaptive Censor stopped. Saved local file sessions remain available.'); return
    url = f'http://127.0.0.1:{args.port}'
    running = False
    try:
        response = httpx.get(url + '/api/bootstrap', timeout=1)
        running = response.status_code == 200 and response.json().get('version') == '0.3.0'
    except (httpx.HTTPError, ValueError):
        pass
    if not running:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        with (root/'server.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('server.py')), '--port', str(args.port)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags,
                                       start_new_session=os.name != 'nt', cwd=Path(__file__).parent)
        record.write_text(json.dumps({'pid':process.pid, 'created':psutil.Process(process.pid).create_time()}), encoding='utf-8')
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(f'The server exited. See {root / "server.log"}')
            try:
                response = httpx.get(url+'/api/bootstrap', timeout=1)
                if response.status_code == 200:
                    running = True; break
            except httpx.HTTPError:
                pass
            time.sleep(.25)
        if not running:
            raise RuntimeError(f'Startup timed out. See {root / "server.log"}')
    print(f'Adaptive Censor is running at {url}')
    if not args.no_browser:
        webbrowser.open(url)


if __name__ == '__main__':
    main()
