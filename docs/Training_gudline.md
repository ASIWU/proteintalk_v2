# Training requirement

The code base in train.py and infer.py should have the training pipeline below: 

## 1. Single drug prediction:

1. 5-fold split on stratified pert_id, cell_type and cell
2. 5-fold split on stratified pert_id without protome(this means just no use mse loss) or without pdi (ablation)

## 2. Double drug :
1. 5-fold split on pert_id

## 3. Extra data :
1. training on all-single data, then test on extra_single data (mat[1-4])
2. training on all-single and double data, then test on extra_double data (nature, nc, guomics)