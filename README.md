[![Stories in Ready](https://badge.waffle.io/d-meiser/ion-trapping-notes.png?label=ready&title=Ready)](https://waffle.io/d-meiser/ion-trapping-notes?utm_source=badge)

# ion-trapping-notes

Some notes for simulations of ultra-cold ions in Penning traps.


## Contents of this repository

* [Notes on Penning trap simulations](cooling/cooling.tex)
* [Notes on how to extract mode temperatures from ion trajectories](cooling/axial_thermometry.tex)
* [Scripts for simulations and creating figures](cooling/scripts)
* [Figures for notes](cooling/figures)
* [Python module with functions for Penning trap simulations](ion_trapping)


## Getting started

Installing the following Python packages should allow you to run the simulation
scripts in `cooling/scripts`:
```{bash}
pip install coldatoms
pip install scipy
pip install .
```


[Notes on managing git repositories](git-forking-notes.md)
