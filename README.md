### Description
Set of files to re-create numerical results published in the following article:

* P.-Y. C. R. Taunay, C. J. Wordingham, E. Y. Choueiri, 
"Total pressure in thermionic orificed hollow cathodes: controlling mechanisms and their 
relative importance," 
Journal of Applied Physics, 2021

#### Experimental data
The experimental data we used are available in the [cathode-database repository](https://github.com/eppdyl/cathode-database) 
(DOI: 10.5281/zenodo.3956853). A copy of the cathode database is available in the HDF5 file format
under a CC-BY-4.0 license.

---
### How to use

#### Re-creating numerical results
To re-create the numerical results, run the scripts that are located in the ./article folder. 
Scripts are organized by paper section. 
Each script has a docstring below the license to provide more context. 
The scripts may be run directly on a given machine or through the provided 
[Singularity](https://sylabs.io/) container.

##### Container 
To ensure reproducibility, a Singularity container is also provided to run the examples.
The container must first be built:

```bash
cd ./container
# If you do not have root access...
singularity build --fakeroot singularity.sif singularity.def
# ...Or if you do
sudo singularity build singularity.sif singularity.def
```

The container environment is now setup and the required Python packages are downloaded. The Python 
scripts can then be run through the container 

```bash
cd ../article/sectionIII_pressure-statistical-analysis/B_power-law-approach/
singularity exec ../../../container/singularity.sif python3 power-law_fit.py
``` 

---
### License
All software files are licensed under MIT license.
All data files are licensed under CC-BY-4.0 license. 

---
Pierre-Yves Taunay, 2021
