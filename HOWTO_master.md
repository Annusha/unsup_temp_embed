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

during testing will be created several folders which by default stored at --dataset_root, change if necessary --output 

> segmentation/  
> likelihood/  
> logs/  

#### Test


##### Test on Breakfast

- Breakfast features [link](https://drive.google.com/open?id=1Ar4XKA_moL7gcczjxKpZZY4zJ_zBcOdG)
- Breakfast ground-truth [link](https://drive.google.com/open?id=1R3z_CkO1uIOhu4y2Nh0pCHjQQ2l-Ab9E)
- pretrained models [link](https://drive.google.com/open?id=1Ok5w5yvDP5VBuaJj1k17J1OWsw-j58_z)
- action: 'coffee', 'cereals', 'tea', 'milk', 'juice', 'sanwich', 'scrambledegg', 'friedegg', 'salat', 
'pancake'  
 &nbsp;&nbsp;&nbsp;&nbsp;use 'all' to run test on all actions in series 
 ```bash
python data_utils/BF_utils/bf_test.py
```
 
 

##### Test on Inria YouTube Instructions

- YouTube Instructions features [link](https://www.di.ens.fr/willow/research/instructionvideos/) 
- YouTube Instructions ground-truth [link](https://drive.google.com/open?id=1IUE_iiEB_5bR5CUk0L4jN9NnQTLuZd4y)
- pretrained models [link](https://drive.google.com/open?id=1Ao_sC9ZPX8ZznCyLNAclkGIE3aBSekIC)
- action: 'changing_tire', 'coffee', 'jump_car', 'cpr', 'repot'  
 &nbsp;&nbsp;&nbsp;&nbsp;use 'all' to run test on all actions in series  

##### Test on 50Salads

- 50Salads features [link]( https://drive.google.com/open?id=1jTEwy-VpuSpB53nwgymVvmXWglFrfK7k)
- 50Salads ground-truth [link](https://drive.google.com/open?id=1pL6MjaWCLFo_jJ4UKjrjgPju2pyQeBxr)
- pretrained model [link](https://drive.google.com/open?id=1mTfm15zC3Uc-_NMApuEiqosaiQUnivzJ)
- action: 'rgb' == 'all'

for 50Salad dataset there is only one model since people make just some variations of the same salad and there is no 
other activity classes


##### Test your own data
see folders dummy_utils and dummy_data and modify files respectively

#### Train your own model
