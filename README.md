# Fast-eTofts

Python code for JMRI paper "A Convolutional Neural Network Forfor Accelerating the Computation of the Extended Tofts Model in DCE-MRI"

## Dataset

1. ##### The original DCE-MRI data and segmentation data are stored in .mat format

   ###### DCE-MRI/mat

```
DCE_XX
├── Cp.mat
├── mask_edema_segment.mat
├── mask_NAWM_segment.mat
├── mask_tumor_segment.mat
├── mask_whole_brain.mat
├── T10.mat
├── DCE
│   ├── dce1.mat  # raw DCE-MRI data
│   ├── dce2.mat
│   ├── ...
│   ├── ...
│   ├── dce118.mat
│   └── gaus  # DCE-MRI data after Gaussian filter
│       ├── gaus_dce1.mat
│       ├── gaus_dce2.mat
│       ├── ...
│       ├── ...
│       ├── gaus_dce118.mat
│       └── whole.npy  # Convert Gaussian filtered data to .npy format
DCE_XX
...
DCE_XX
```

###### Use multi-process to convert .mat format to .npy format for model training and testing

```
python3 from_mat_to_npy.py --mat=DCE-MRI/mat/DCE_XX  --npy=DCE-MRI/train
```

```
npy file data path: 

	DCE-MRI/train.patientXX
```

2. #####  Train and test dataset path file should be organized in the following structure

   ###### DCE-MRI/train(test)

```
├── patientXX
│   ├── cp.npy
│   ├── normal_1.npy
│   ├── normal_2.npy
│   ├── tumor_1.npy
│   ├── tumor_2.npy
│   └── tumor_3.npy
├── patientXX
├── ...
├── ...
├── patientXX
```



## Environment

```
python3.6
```

###### pip requirements.txt

```
toml==0.10.2
matplotlib==3.1.1
easydict==1.9
scipy==1.5.2
torch==1.1.0
numpy==1.16.4
```

## Train and test

###### Train model

Example:

```
$ python3 train.py --gpu=0 --strategy=mix
```

###### train.py usage

```
Usage:
    python3 train.py [options]

Options:
    --gpu            Int, Specify the graphics device, the default is 0
    --weight   	     Int, fit loss weight, default 40
    --stop_num       Early stop number, default 5, 
    --lr             Initial learning rate, default 0.0005
    --name           Experiment name, default "train"
    --strategy       Training strategy, select from  'synthetic', 'patient', 'fine_tune' or 'mix'
Return: 
	Generate a file named
	"exp/$name$/$strategy$_$weight$_$stop_num$_&lr&_$datatime$/
	train_result_patientXX.txt"
	
```

###### Test model

```
$ python3 test.py --gpu=0 --model_patah=exp/train/mix/best_patient.tar
```

###### test.py usage

```
Usage:
    python3 test.py [options]

Options:
    --gpu            Int, Specify the graphics device, the default is 0
    --model_path     File or directory path, Specify the path of model  parameter.  If a directory, the complete model path will be completed as "$model_path$/best_patient.tar"
Return: 
	Generate files named
	"$model_path$/test_result_patientXX.txt"
```

###### The content of the result file is 

```
k_trans,$MAE$,$CCC$,$NRMSE$,vb,$MAE$,$CCC$,$NRMSE$,ve,$MAE$,$CCC$,$NRMSE$,resx,$NLLS FIT RESIDUAL$,resy,$CNN FIT RESIDUAL$
```

## License

All Rights Reserved.

