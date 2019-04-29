# Unsupervised learning of action classes with continuous temporal embedding

Official implementation in python.  https://arxiv.org/abs/1904.04189

#### One activity class

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/local_pipeline_v.png)


##### Create environment
```
conda create --name cte --file requirements.txt
```

##### Input files
```
one file per video
# rows = # frames in video
# columns = dimensionality of frame-wise features
```
to extract frame-wise features use improved dense trajectories (this step can be substituted by smth else)


##### Run your own data
see folders TD_utils and test_data and modify files respectively

