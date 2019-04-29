# Unsupervised learning of action classes with continuous temporal embedding

Official implementation in python.  https://arxiv.org/abs/1904.04189

If you use the code, please cite


```
A. Kukleva, H. Kuehne, F. Sener , and J. Gall:
Unsupervised Learning of Action Classes with Continuous Temporal Embedding 
in IEEE Conference on Computer Vision and Pattern Recognition (CVPR'19)
```

Two branches: master [(howto)](https://github.com/Annusha/unsup_temp_embed/blob/master/HOWTO.md), global [(howto)](https://github.com/Annusha/unsup_temp_embed/blob/global/HOWTO.md)

master: 
Pipeline for one activity class. Figure 1 in the paper.

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/local_pipeline_v.png)

global:
Proposed pipeline for unsupervised learning with unknown activity classes. Figure 2 in the paper.

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/global_pipeline_v.png)


#### Visualization of embeddings via t-SNE on the 50Salads dataset

Each frame is color coded a) with the corresponding ground truth subaction label, b) with K assigned subaction labels after clustering as the second step in Fig.1 in our main paper, c)with the predicted labels after the decoding stage. The optimization of our network is performed with respect to relative timestep of each frame. In d) we show the respective relative time label in the continuous temporal embedding assigned to each frame feature. The color bar depicts that bright blue corresponds to 0 (startof the video) and pink to 1 (end of the video).

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/embedding.png)


#### The number of subactions K

##### Breakfast dataset

| Activity class name  | # subactions (K) |
| -------------------- | ---------------- |
|        Coffe         |        7         |
|        Cereals       |        5         |
|        Tea           |        7         |
|        Milk          |        5         |
|        Juice         |        8         |
|        Sandwich      |        9         |
|        Scrambledegg  |       12         |
|        Friedegg      |        9         |
|        Salat         |        8         |
|        Pancake       |       14         |

##### YouTube Instractions dataset

| Activity class name  | # subactions (K) |
| -------------------- | ---------------- |
|        Changing tire |       11         |
|        Making cofee  |       10         |
|        CPR           |        7         |
|        Jump car      |       12         |
|        Repot plant   |        8         |


#### Qualitative results

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/cereals.png)
Breakfast dataset. The order of subactions: SIL, take bowl, pour cereals, pour milk, stir cereals, SIL

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/tea.png)
Breakfast dataset. The order of subactions: SIL, take cup, add teabag, pour water, SIL

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/milk.png)
Breakfast dataset. The order of subactions: SIL, spoon powder, pour milk, stir milk, SIL

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/juice.png)
Breakfast dataset. The order of subactions: SIL, take knife, cut orange, squeeze orange, pour juice, squeeze orange, pour juice, squeeze orange, pour juice, squeeze orange, pour juice, SIL

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/sandwich.png)
Breakfast dataset. The order of subactions: SIL, cut bun, smear butter, put toppingOnTop, SIL

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/rgb-01-1_frames.png)
50Salads dataset. The order of subactions: start, cut, place, cut, place, cut, place, cut, place, null, null, add oil, add pepper, mix dressing, end

![alt text](https://github.com/Annusha/unsup_temp_embed/blob/master/supp_mat/rgb-25-2_frames.png)
50Salads dataset. The order of subactions: start, cut, place, cut, place, cut, place, peel cucumber, cut, place, mix ingredients, add oil, null, add pepper, null, mix dressing, serve salad onto plate, add dressing, end



##### TODO descriptions

- [ ] Evaluation
- [ ] Reproduce numbers
- [x] Qualitative results
- [ ] Dense trajectorues extaction
- [ ] table 1, videovector howto  


