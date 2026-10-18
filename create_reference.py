"""Create a harmless export-parity reference. Its patterns are not accuracy labels."""
from labeling import project_dir, read_json, write_json
from analysis_engine import media_info
from core import make_demo
from model_setup import prepare_environment


def main():
    root=prepare_environment();folder=project_dir(root,'model-reference')
    source=folder/'clips'/'patterns.mp4'
    make_demo(source);info=media_info(source)
    manifest=folder/'project.json'
    if not manifest.exists():
        write_json(manifest,dict(version=1,name='model-reference',context=2,purpose='Synthetic feature export parity only',windows=[
            dict(id='patterns',title='Generated patterns',episode='demo',split='dev',stratum='synthetic',
                 clip_offset=0,clip_duration=info['duration'],has_audio=info['has_audio'],fps=24,target=[2,10])]))
    print(folder)


if __name__=='__main__':main()
