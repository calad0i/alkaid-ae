import sys
from glob import glob
from subprocess import run

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit()
    print('synth time: ', end='')
    for root in sys.argv[1:]:
        p0 = f'{root}/synth.log'
        p1 = glob(f'{root}/output_*/reports/*_post_synth_timing.rpt')[0]
        t0 = run(f'stat -c %W {p0}', shell=True, capture_output=True, text=True).stdout.strip()
        t1 = run(f'stat -c %W {p1}', shell=True, capture_output=True, text=True).stdout.strip()
        print(f'{(int(t1) - int(t0)) / 60:.2f} mins', end=' ')
    print()
