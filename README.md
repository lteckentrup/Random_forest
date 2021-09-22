# Random_forest

1. Set up on HPC. First build virtual environment:

```
module purge
rm -r /home/561/lt0205/.local/lib/python3.9
module load tensorflow/2.6.0
python3 -mvenv --system-site-packages my-venv
my-venv/bin/pip3 install --upgrade pip
my-venv/bin/pip3 install --no-binary :all: pandas
my-venv/bin/pip3 install --no-binary :all: tensorflow
my-venv/bin/pip3 install tensorflow_decision_forests
my-venv/bin/pip3 install --no-binary :all: wurlitzer
module load netcdf/4.8.0
module load hdf5/1.12.1
my-venv/bin/pip3 install --no-binary :all: netCDF4
```
then load rest of modules

```
module load pbs
module load python3/3.9.2
module load cuda/11.4.1
module load cudnn/8.2.2-cuda11.4
module load nccl/2.10.3-cuda11.4
module load openmpi/4.1.1
export LD_PRELOAD=/apps/openmpi/4.1.1/lib/libmpi_cxx.so
```
activate environment
```
source my-venv/bin/activate
```
and run script
```
python3 pixel_test.py 
```

