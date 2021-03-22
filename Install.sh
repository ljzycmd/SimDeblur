# install packages in requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Compile and install deformable convolution for EDVR
cd ./simdeblur/model/backbone/edvr/dcn
python setup.py clean
python setup.py develop
cd -
# compile and install FAC layer in STFAN
cd ./simdeblur/model/backbone/stfan/FAC/kernelconv2d
python setup.py clean
python setup.py develop
# 

cd -
# install the SimDeblur
python setup.py develop