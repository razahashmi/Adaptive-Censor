"""Small Windows CPU setup helper. Runtime dependencies/models download separately.

This module uses only the standard library and the source-archive verifier.
The frozen executable verifies its adjacent archive against an embedded hash.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import stat
import struct
import subprocess
import sys
import threading
import webbrowser
import zipfile

from build_source_release import verify

REPOSITORY='https://github.com/razahashmi/Adaptive-Censor'
FLAGS=getattr(subprocess,'CREATE_NO_WINDOW',0)

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def environment(runtime=None):
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    if runtime:env['ADAPTIVE_CENSOR_WORKDIR']=str(runtime)
    return env

def inspect_python(command):
    try:
        result=subprocess.run([*command,'-c','import sys,struct,json; print(json.dumps(dict(version=list(sys.version_info[:2]),bits=struct.calcsize("P")*8,executable=sys.executable)))'],
            capture_output=True,text=True,timeout=15,env=environment(),creationflags=FLAGS)
        info=json.loads(result.stdout)
        if result.returncode==0 and info['version']==[3,13] and info['bits']==64:return [info['executable']]
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired):pass
    return None

def find_python(explicit=None):
    commands=[[str(explicit)]] if explicit else [['py','-3.13'],['python']]
    for command in commands:
        found=inspect_python(command)
        if found:return found
    raise RuntimeError('Install 64-bit Python 3.13, then reopen this setup helper.')

def check_archive(archive,expected_hash=None):
    archive=Path(archive)
    if expected_hash and digest(archive)!=expected_hash:raise ValueError('The source archive does not match the checksum embedded in this setup build.')
    return verify(archive)

def reject_links(path):
    path=Path(path)
    for part in (path,*path.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0)):
            raise ValueError('Choose an ordinary folder without links or junctions.')

def extract_source(archive,destination,expected_hash=None):
    info=check_archive(archive,expected_hash)
    destination=Path(destination).expanduser().absolute();reject_links(destination)
    app=destination/info['version'];reject_links(app)
    with zipfile.ZipFile(archive) as z:
        prefix=f"adaptive-censor-{info['version']}/"
        values={name[len(prefix):]:z.read(name) for name in z.namelist()}
    if app.exists():
        # Reopening is safe; an existing installation or source edit is never reset.
        for name,data in values.items():
            p=app/name;reject_links(p)
            if not p.is_file() or p.read_bytes()!=data:raise ValueError('This version folder differs from the setup archive. Choose another installation folder; no file was overwritten.')
        return app,info
    app.mkdir(parents=True)
    for name,data in values.items():
        p=app/name
        if not p.resolve().is_relative_to(app.resolve()):raise ValueError('External extraction target')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return app,info

def execute(command,app,log,status):
    with log.open('ab') as stream:
        result=subprocess.run(command,cwd=app,env=environment(app/'.runtime'),stdin=subprocess.DEVNULL,
                              stdout=stream,stderr=stream,creationflags=FLAGS)
    if result.returncode:raise RuntimeError('Setup could not complete this step. Read setup.log in the installation folder and use the installation issue form for help.')

def install(archive,destination,python=None,models=False,status=lambda value:None,expected_hash=None):
    command=find_python(python);status('Verifying and preparing source files…')
    app,info=extract_source(archive,destination,expected_hash);log=app/'setup.log'
    venv=app/'.venv';runtime=venv/'Scripts/python.exe'
    if not runtime.exists():
        status('Creating an isolated Python environment…');execute([*command,'-m','venv',str(venv)],app,log,status)
    reject_links(venv);reject_links(runtime)
    if not inspect_python([str(runtime)]):raise RuntimeError('The installation environment needs 64-bit Python 3.13.')
    status('Installing pinned CPU dependencies. This needs an internet connection…')
    execute([str(runtime),'-m','pip','install','--require-virtualenv','--disable-pip-version-check','-r',str(app/'requirements-cpu-win-py313.lock.txt')],app,log,status)
    if models:
        status('Downloading the separate base speech and visual models…')
        execute([str(runtime),str(app/'model_setup.py')],app,log,status)
    status('Checking the installation…')
    result=subprocess.run([str(runtime),str(app/'readiness.py')],cwd=app,env=environment(app/'.runtime'),capture_output=True,text=True,creationflags=FLAGS,timeout=60)
    readiness=json.loads(result.stdout)
    if result.returncode or not readiness['demo_ready']:raise RuntimeError('Installation checks found a problem. Open Check setup or inspect setup.log.')
    status('Ready. The controls demo works without separate model downloads.' if not models else 'Ready. Open the player to review your own media.')
    return dict(app=str(app),version=info['version'],source_sha256=info['sha256'],models_requested=models,demo_ready=readiness['demo_ready'],
                clean_machine_validated=False,os_reinstalled=False)

def start(app,port=8502,browser=True):
    app=Path(app);runtime=app/'.venv/Scripts/python.exe'
    command=[str(runtime),str(app/'launcher.py'),'start','--port',str(port)]
    if not browser:command.append('--no-browser')
    result=subprocess.run(command,cwd=app,env=environment(app/'.runtime'),capture_output=True,text=True,creationflags=FLAGS,timeout=45)
    if result.returncode:raise RuntimeError('The player did not start. See .runtime/server.log in your installation folder.')
    return 'http://127.0.0.1:'+str(port)

def build_info():
    path=Path(getattr(sys,'_MEIPASS',Path(__file__).parent))/'setup-build-info.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}

def gui(archive,expected_hash=None):
    import tkinter as tk
    from tkinter import ttk,filedialog,messagebox
    window=tk.Tk();window.title('Adaptive Censor · CPU setup');window.geometry('650x420');window.resizable(False,False)
    ttk.Style().theme_use('clam')
    frame=ttk.Frame(window,padding=26);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Your media. Your rules.',font=('Segoe UI',22,'bold')).pack(anchor='w')
    ttk.Label(frame,text='Set up the local Windows CPU preview. Python 3.13 is required.\nDependencies and optional models download separately.',wraplength=580).pack(anchor='w',pady=(12,16))
    folder=tk.StringVar(value=str(Path(os.environ.get('LOCALAPPDATA',Path.home()))/'AdaptiveCensor'))
    row=ttk.Frame(frame);row.pack(fill='x');ttk.Entry(row,textvariable=folder,width=59).pack(side='left')
    ttk.Button(row,text='Choose…',command=lambda:folder.set(filedialog.askdirectory(initialdir=folder.get()) or folder.get())).pack(side='right')
    models=tk.BooleanVar(value=False)
    ttk.Checkbutton(frame,text='Also download base models for analyzing my own files',variable=models).pack(anchor='w',pady=(16,8))
    ttk.Label(frame,text='Unchecked: controls demo only. GPU and scene-verifier setup are separate.').pack(anchor='w')
    state=tk.StringVar(value='Choose a folder, then select Set up.');ttk.Label(frame,textvariable=state,wraplength=580).pack(anchor='w',pady=20)
    messages=queue.Queue();installed=[None]
    def worker(destination,with_models):
        try:
            result=install(archive,destination,models=with_models,status=lambda s:messages.put(('status',s)),expected_hash=expected_hash)
            messages.put(('ready',result['app']))
        except Exception as error:messages.put(('error',str(error)))
    def setup():
        setup_button.configure(state='disabled');threading.Thread(target=worker,args=(folder.get(),models.get()),daemon=True).start()
    def open_player():
        try:start(installed[0])
        except Exception as error:messagebox.showerror('Player could not start',str(error))
    buttons=ttk.Frame(frame);buttons.pack(fill='x')
    setup_button=ttk.Button(buttons,text='Set up',command=setup);setup_button.pack(side='left')
    open_button=ttk.Button(buttons,text='Open player',state='disabled',command=open_player);open_button.pack(side='left',padx=10)
    ttk.Button(buttons,text='Get Python 3.13',command=lambda:webbrowser.open('https://www.python.org/downloads/windows/')).pack(side='right')
    ttk.Button(frame,text='Installation guide / help',command=lambda:webbrowser.open(REPOSITORY+'/blob/main/docs/INSTALLATION.md')).pack(anchor='w',pady=(16,0))
    def poll():
        try:
            while True:
                kind,value=messages.get_nowait()
                if kind=='status':state.set(value)
                elif kind=='ready':installed[0]=value;open_button.configure(state='normal');setup_button.configure(state='normal')
                else:state.set(value);setup_button.configure(state='normal')
        except queue.Empty:pass
        window.after(150,poll)
    window.after(150,poll);window.mainloop()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path);p.add_argument('--destination',type=Path)
    p.add_argument('--python',type=Path);p.add_argument('--install-demo',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--report',type=Path);p.add_argument('--start',action='store_true');p.add_argument('--port',type=int,default=8502);p.add_argument('--no-browser',action='store_true')
    p.add_argument('--expected-sha256');p.add_argument('--download-models',action='store_true')
    args=p.parse_args();info=build_info()
    archive=args.archive or Path(sys.executable if getattr(sys,'frozen',False) else __file__).parent/info.get('archive','adaptive-censor-0.3.0-preview.1-source.zip')
    expected=info.get('sha256') or args.expected_sha256
    if args.self_test:
        result=dict(**check_archive(archive,expected),interpreter_found=bool(find_python(args.python)),gui_tested=False,network_requests=False)
    elif args.install_demo:
        if not args.destination:p.error('--install-demo requires --destination')
        result=install(archive,args.destination,args.python,args.download_models,expected_hash=expected)
        if args.start:result['url']=start(result['app'],args.port,not args.no_browser)
    else:gui(archive,expected);return
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if sys.stdout:print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if getattr(sys,'frozen',False) and not any(a in sys.argv for a in ('--self-test','--install-demo')):
            import tkinter.messagebox
            tkinter.messagebox.showerror('Adaptive Censor setup',str(error))
            raise SystemExit(1)
        else:raise
