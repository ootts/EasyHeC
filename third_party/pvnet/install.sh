cd lib/csrc
#export CUDA_HOME="/usr/local/cuda"
cd ransac_voting
/bin/rm -r build *so
python setup.py build_ext --inplace
cd ../nn
/bin/rm -r *so
python setup.py build_ext --inplace
cd ../fps
/bin/rm -r *so
python setup.py build_ext --inplace
cd ../../..

# If you want to run PVNet with a detector
#cd ../dcn_v2
#python setup.py build_ext --inplace

# If you want to use the uncertainty-driven PnP

#cd lib/csrc
#export CUDA_HOME="/usr/local/cuda-10.2"
#cd ransac_voting
#python setup.py build_ext --inplace
#cd ../nn
#python setup.py build_ext --inplace
#cd ../fps
#python setup.py build_ext --inplace