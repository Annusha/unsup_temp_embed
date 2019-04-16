# Unsupervised learning of action classes with continuous temporal embedding

Official implementation in python.  https://arxiv.org/abs/1904.04189

Two branches: master, global

master: 
Pipeline for one activity class. Figure 1 in the paper.

global:
Proposed pipeline for unsupervised learning with unknown activity classes. Figure 2 in the paper.


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


##### TODO descriptions

- [ ] Evaluation
- [ ] Reproduce numbers
- [ ] Qualitative results
- [ ] Dense trajectorues extaction
- [ ] table 1, videovector howto  

