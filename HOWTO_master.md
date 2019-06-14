# Unsupervised learning of action classes with continuous temporal embedding

Official implementation in python.  https://arxiv.org/abs/1904.04189

#### One activity class

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/local_pipeline_v.png)


##### Create environment
```
conda create --name ute --file requirements.txt
conda actiavte ute
```
install torch=1.0 for your machine

##### Input files
```
one file per video
# rows = # frames in video
# columns = dimensionality of frame-wise features
```
to extract frame-wise features use improved dense trajectories (this step can be substituted by smth else)

###### General parameters for training and testing
```
--dataset_root=/path/to/your/root/data/folder/
--action='coffee'  # dataset dependant, see below
# set feature extension and dimensionality
--ext = 'txt | tar.gz | npy'
--feature_dim=64 

 # default is 'mlp', set 'nothing' for no embedding (just raw features)
--model_name='mlp'


# resume training
--resume=True | False  
--load_model=True | False
# if load_model == True then specify
--loaded_model_name='name_of_your_model'

# if dataset has background class (like YTI) set to True 
--bg=False
```
set parameters as arguments for the script or update them in the corresponding one


#### Folder structure
for each dataset create separate folder (specify path --dataset_root) where the inner folders structure is as following:

> features/  
> groundTruth/  
> mapping/  
> models/

during testing will be created several folders which by default stored at --dataset_root, change if necessary 
--output_dir 

> segmentation/  
> likelihood/  
> logs/  

#### Reproduce numbers


##### Test on Breakfast

- Breakfast features [link](https://drive.google.com/open?id=1Ar4XKA_moL7gcczjxKpZZY4zJ_zBcOdG)
- Breakfast ground-truth [link](https://drive.google.com/open?id=1-1ie5gAwQozhIHr_ggMtlxgrE699dnvc)
- pretrained models [link](https://drive.google.com/open?id=1Ok5w5yvDP5VBuaJj1k17J1OWsw-j58_z)
- actions: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 
'pancake'  
 &nbsp;&nbsp;&nbsp;&nbsp;use 'all' to run test on all actions in series 
 ```bash
 # to test pretrained models
python data_utils/BF_utils/bf_test.py

# to train models from scratch
python data_utils/BF_utils/bf_train.py
```
 
 

##### Test on Inria YouTube Instructions

- YouTube Instructions features [link](https://drive.google.com/open?id=1HyF3_bwWgz1QNgzLvN4J66TJVsQTYFTa) 
- YouTube Instructions ground-truth [link](https://drive.google.com/open?id=1ENgdHvwHj2vFwflVXosCkCVP9mfLL5lP)
- pretrained models [link](https://drive.google.com/open?id=1LRRCfFTKzY4cXCQiTnMnG_qRxhQ0EkjR)
- actions: 'changing_tire', 'coffee', 'jump_car', 'cpr', 'repot'  
 &nbsp;&nbsp;&nbsp;&nbsp;use 'all' to run test on all actions in series  
 
 ```bash
 # to test pretrained models
python data_utils/YTI_utils/yti_test.py

# to train models from scratch
python data_utils/YTI_utils/yti_train.py
```
 

##### Test on 50Salads

- 50Salads features [link](https://drive.google.com/open?id=17o0WfF970cVnazrRuOWE92-OiYHEXTT3)
- 50Salads ground-truth [link](https://drive.google.com/open?id=1mzcN9pz1tKygklQOiWI7iEvcJ1vJfU3R)
- pretrained model [link](https://drive.google.com/open?id=1mTfm15zC3Uc-_NMApuEiqosaiQUnivzJ)
- actions: 'rgb' == 'all'

for 50Salad dataset there is only one model since people make just some variations of the same salad and there is no 
other activity classes

 ```bash
 # to test pretrained models
python data_utils/FS_utils/fs_test.py

# to train models from scratch
python data_utils/FS_utils/fs_train.py
```


#### Dummy data (template for training and testing your own data)
 ```bash
 # to test pretrained model
python data_utils/dummy_utils/dummy_test.py

# to train model from scratch
python data_utils/dummy_utils/dummy_train.py
```
see folders 
> dummy_data/  
> data_utils/dummy_utils/    

and modify with respect to your own data

