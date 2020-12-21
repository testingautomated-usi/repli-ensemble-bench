# Lazy Ensemble Benchmark (Reproduction Package)

This repository contains the reproduction of our performance benchmarks (table 1)
in the paper: *Uncertainty Wizard; Fast and User-Friendly Neural Network Uncertainty Quantification*
(preprint available upon request).

For training, the scripts `run_in_ubu_2xgpu.py` and `run_on_win_2080_ti.py` were used.
So created artefacts were manually moved into an according subfolder in `results/`.
The final results (as shown an the paper and in this readme below) can be extracted using `table_plotter.py`

On the ubuntu machine, the training scripts were executed in a docker container, created using the `Dockerfile`
in this repository.
On the windows machine, a python 3.6 (64 bit) environment was used.

For questions regarding `uncertainty-wizard` please refer to the 
[library repository](https://github.com/testingautomated-usi/uncertainty-wizard).

## Results 

### PC: Dual-GPU custom-built Ubuntu20.04
CPU = Threadripper 1920X, GPU 0 = GTX 1060, GPU 1 = GTX 1070Ti


Context: mainprocess
>	Total training time: 5h 10min<br>
>	Avg load CPU: 8.9%<br>
>	Avg load GPU 0: 0.0%<br>
>	Avg load GPU 1: 45.42%<br>

Context: multiprocess
>	Total training time: 2h 57min<br>
>	Avg load CPU: 14.28%<br>
>	Avg load GPU 0: 0.09%<br>
>	Avg load GPU 1: 92.24%<br>

Context: multigpu
>	Total training time: 1h 41min<br>
>	Avg load CPU: 26.77%<br>
>	Avg load GPU 0: 98.19%<br>
>	Avg load GPU 1: 92.41%<br>


### PC: Alienware Aurora R8
CPU = i7 9700, GPU 0 = RTX 2080Ti 

Context: mainprocess
>	Total training time: 3h 7min<br>
>	Avg load CPU: 14.74%<br>
>	Avg load GPU 0: 46.93%<br>

Context: multiprocess
>	Total training time: 2h 32min<br>
>	Avg load CPU: 50.31%<br>
>	Avg load GPU 0: 83.9%<br>



