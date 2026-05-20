Now we focus on PTV1. I found that although the final target and format is same, but PTV1 have many different process workflow. Please follow the below requirements: 
1) i split the ptv1 data in rad data folder to 2 parts : (1)  ptv1 and (2) ptv1_extra_singledrug. So please process them like ptv3 and ptv3 extra data
for ptv1,
2)  all data is record in the aivc.csv including the info and the protein expression, i obtain the information to aivc_info.csv, 
3) the find control logic is diff to the ptv3: (1) only the protein_plate, BioRep is same and the pert_time is 0 is the control, which mean samples with pert_time not equals to 0 is pert data.
4) convert the "NY_label" columns to PRISM1st_label_total
5) the smiles is not record in aivc.csv, which is in ptv1.csv. you should read and understand the ptv1.csv to obtain the smiles and target feature.
for ptv1_extra_singledrug
6) first, you can get samples' ground_truth, cell and E115_id from the test12091214_sample_predictions_E115id.csv
7) then, you can use cell to get the baseline proteome from the aivc.csv, in which you needs the same value for cell in test12091214_sample_predictions_E115id.csv and cell_plate in aivc.csv 
8) next, E115_id is the pert_id. so you can get smiles and target from **`PTV3` global_meta.json**