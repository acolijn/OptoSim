##!/bin/sh

python generate_jobs.py --mcid=0 --njobs=100 --config=config_nev1k_nph100k.json
python generate_jobs.py --mcid=1 --njobs=100 --config=config_nev10k_nph10k.json
python generate_jobs.py --mcid=2 --njobs=100 --config=config_nev10k_nph10k_noscatter.json
python generate_jobs.py --mcid=3 --njobs=100 --config=config_nev10k_nph10k_nodiffuse.json
python generate_jobs.py --mcid=4 --njobs=100 --config=config_nev10k_nph10k.json
python generate_jobs.py --mcid=5 --njobs=100 --config=config_nev10k_nph10k_noscatter.json
