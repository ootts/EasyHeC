## 1. Install EasyHeC.

   ```bash
   # Clone repo
   git clone https://github.com/haosulab/easyhec.git --recursive
   conda create -n easyhec python=3.7
   conda activate easyhec
   # Install Pytorch with the following command or follow the official doc at https://pytorch.org/get-started/locally/.
   conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
   # Install other dependencies
   conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
   conda install pytorch3d -c pytorch3d -y
   pip install -r requirements.txt
   # Build
   sh build.sh
   ```

## 2. Install third parties.

### a). Install pvnet.

```bash
cd third_party/pvnet
sudo apt-get install libglfw3-dev libglfw3
# pip install -r requirements.txt
sh install.sh
cd ../..
```

### b). Install detectron2.

```bash
cd third_party/detectron2
pip install -e .
cd ../..
```

## 3. Install Xarm-Python-SDK.

```bash
cd third_party/xArm-Python-SDK
python setup.py install
cd ../..
```



