#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH -e slurm-%j.out
#SBATCH -o slurm-%j.out
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

source ~/.snana

date

cd /global/cscratch1/sd/kessler/SNANA_LSST_SIM/AGAG_TEXTRUNS_20k
activate superraenn

superraenn-extract products/lcs_2022-01-22.npz

date
