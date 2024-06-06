# Random Policy
```python
python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_easy_random.yaml --dataset_dir datasets/ 
python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_medium_random.yaml  --dataset_dir datasets/ python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_hard_random.yaml  --dataset_dir datasets/
```

![[PointmassHard-v0_random.png]]
![[PointmassMedium-v0_random.png]]
![[PointmassHard-v0_random.png]]
# RND Policy
```python
python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_easy_rnd.yaml --dataset_dir datasets/ 
python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_medium_rnd.yaml  --dataset_dir datasets/ 
python cs285/scripts/run_hw5_explore.py  -cfg experiments/exploration/pointmass_hard_rnd.yaml  --dataset_dir datasets/
```

![[PointmassEasy-v0_rnd1.png]]
![[PointmassMedium-v0_rnd1.png]]
![[PointmassHard-v0_rnd1.png]]
# CQL
```python
python ./cs285/scripts/run_hw5_offline.py  -cfg experiments/offline/pointmass_easy_cql.yaml  --dataset_dir datasets 
python ./cs285/scripts/run_hw5_offline.py  -cfg experiments/offline/pointmass_medium_cql.yaml  --dataset_dir datasets 
python ./cs285/scripts/run_hw5_offline.py  -cfg experiments/offline/pointmass_easy_dqn.yaml  --dataset_dir datasets 
python ./cs285/scripts/run_hw5_offline.py  -cfg experiments/offline/pointmass_medium_dqn.yaml  --dataset_dir datasets
```

![[CQL.png]]

# AWAC
```python

```
