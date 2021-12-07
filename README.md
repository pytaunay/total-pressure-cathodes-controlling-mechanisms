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
A docstring is provided at the top of each script (below the license) to provide more context. 
The scripts may be run natively or through the provided [Singularity](https://sylabs.io/) container.

##### Container 
To ensure reproducibility, a Singularity container is also provided to run the examples.
The container must be built first:

```bash
cd ./article
singularity build --fakeroot singularity.sif singularity.def
```

The environment is setup and the required Python packages are downloaded. The Python scripts can 
then be run directly

```bash
singularity exec singularity.sif python3 power-law_fit.py
``` 

---
### License
All software files are licensed under MIT license.
All data files are licensed under CC-BY-4.0 license. 

---
Pierre-Yves Taunay, 2021
