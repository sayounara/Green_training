# Green-training https://zenodo.org/badge/269275353.svg
Green-training is a framework integrated with TensorFlow for operation scheduling and energy efficiency optimization on the server architecture with a combination of GPUs and FPGAs without losing performance. It is developed based on tensorflow by adding the FPGA extension to support the schduling and running the operations on both FPGA and GPU.

## Step #1: set up the following hardware and software environments
### Hardware
- Nvidia Tesla V100 GPU
- Intel Stratix 10 FPGA
- A server with two eight-core 2.4 GHz Intel Xeon E5-2630 CPUs
- V100 GPUs and Stratix 10 FPGAs are attached to the server by PCIe 3.0

### Software
- Python 3.5.2, gcc-4.9, g++-4.9, pip3
- Tensorflow 1.8  (https://github.com/tensorflow/tensorflow)
- cuda 9.0, cudnn v7, bazel release 0.10.0
- Intel FPGA SDK for OpenCL version 19.1.0, Inter Quartus Prime version 19.1.0
- Other libraries, e.g., clBLAS 2.11, clDNN (https://github.com/intel/clDNN)
- Ubuntu 16.04


## Step #2: using  Intel FPGA SDK for OpenCL to synthesize the opencl kernels of the offloaded operations to generate the corresponding FPGA bitstreams (this synthesis process could be time-consuming and error-prone)
```
cd Tensorflow_Green_training/tensorflow
\\synthesize each kernel file to get the bitstream file (i.e., aocx file)
aoc cl_kernels/[kernel_name].cl -o fpga_bitsteams/[kernel_name].aocx -board=s10_gh1e1_4Gx2 -v -report
```
## Step #3: using g++-4.9 to generate dynamic link libraries of the FPGA extension for fpga kernels.
```
cd Tensorflow_Green_training/tensorflow/fpga_runtime_src
//for example, compile the source code to get the fpga library for kernel matmul
g++-4.9 -std=c++11 -shared MatConvFPGA.cpp -o libMatConvFPGA.so -fPIC -I~/intelFPGA_pro/19.1/hld/host/include/ ~/intelFPGA_pro/19.1/hld/board/de10_pro/tests/common/src/AOCLUtils/opencl.cpp -I~/intelFPGA_pro/19.1/hld/board/de10_pro/tests/common/inc -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L~/intelFPGA_pro/19.1/hld/host/linux64/lib -lOpenCL
mv *.so fpga_libs
```
## Step #4: using bazel to build the modified Tensorflow to get the tensorflow software installation package and install the package
```
cd Tensorflow_Green_training/tensorflow
./configure
bazel build--config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --action_env=LD_LIBRARY_PATH=/home/jliu/intelFPGA_pro/19.1/hld/host/linux64/lib:${LD_LIBRARY_PATH}
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip3 install -U --user /tmp/tensorflow_pkg/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl
```
## Step #5: running the DNN model benchmarks to test on the platform for training
#### Test resnet50, inception3, vgg16, and alexnet
```
git clone https://github.com/tensorflow/benchmarks
cd benchmarks/scripts/tf_cnn_benchmarks
git checkout cnn_tf_v1.8_compatible
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=~/Tensorflow_Green_training/fpga_libs:${LD_LIBRARY_PATH}
//--model: Model to use, e.g. resnet50, inception3, vgg16, and alexnet.
//train the model with imagenet data set
python3 tf_cnn_benchmarks.py --num_gpus=1 --batch_size=54 --model=resnet50 --variable_update=parameter_server
```
#### Test DCGAN and LSTM
For DCGAN
```
git clone https://github.com/carpedm20/DCGAN-tensorflow
cd DCGAN-tensorflow
//download the dataset
python3 download.py mnist celebA
//To train a model with MNIST dataset
python3 main.py --dataset mnist --input_height=28 --output_height=28 --batch_size=64 --train
```
For LSTM
```
git clone https://github.com/sherjilozair/char-rnn-tensorflow
cd char-rnn-tensorflow
python3 train.py --batch_size=64
```
using NVIDIA System Management Interface, Intel’s Running  Average  Power  Limit  (RAPL)  Interface,  and Intel Powerplay Analyzer  to measure the power consumption of GPU, CPU and FPGA respectively.

