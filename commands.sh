
##########################################
############# Pretrain SSL ###############
##########################################

sbatch sbatch_pretrain_ssl.sh moco
sbatch sbatch_pretrain_ssl.sh barlowtwins 

##########################################
########### Train Classifier #############
##########################################

# note: the output of pretrained model should be present before classifiers can be trained

# train classifier using pretrained moco 
sbatch sbatch_train_clf.sh non 0.0 moco

sbatch sbatch_train_clf.sh sym 0.1 moco
sbatch sbatch_train_clf.sh sym 0.2 moco
sbatch sbatch_train_clf.sh sym 0.3 moco
sbatch sbatch_train_clf.sh sym 0.4 moco
sbatch sbatch_train_clf.sh sym 0.5 moco
sbatch sbatch_train_clf.sh sym 0.6 moco
sbatch sbatch_train_clf.sh sym 0.7 moco
sbatch sbatch_train_clf.sh sym 0.8 moco
sbatch sbatch_train_clf.sh sym 0.9 moco

sbatch sbatch_train_clf.sh asym 0.1 moco
sbatch sbatch_train_clf.sh asym 0.2 moco
sbatch sbatch_train_clf.sh asym 0.3 moco
sbatch sbatch_train_clf.sh asym 0.4 moco
sbatch sbatch_train_clf.sh asym 0.5 moco
sbatch sbatch_train_clf.sh asym 0.6 moco
sbatch sbatch_train_clf.sh asym 0.7 moco
sbatch sbatch_train_clf.sh asym 0.8 moco
sbatch sbatch_train_clf.sh asym 0.9 moco

# train classifier using pretrained barlowtwins 
sbatch sbatch_train_clf.sh non 0.0 barlowtwins   

sbatch sbatch_train_clf.sh sym 0.1 barlowtwins   
sbatch sbatch_train_clf.sh sym 0.2 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.3 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.4 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.5 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.6 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.7 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.8 barlowtwins  
sbatch sbatch_train_clf.sh sym 0.9 barlowtwins  

sbatch sbatch_train_clf.sh asym 0.1 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.2 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.3 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.4 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.5 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.6 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.7 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.8 barlowtwins  
sbatch sbatch_train_clf.sh asym 0.9 barlowtwins  


##########################################
########### Train Classifier #############
###########      NO SSL      #############
##########################################

# train classifier without SSL
# sbatch sbatch_train_clf.sh noise_type noise_rate model_type num_epochs
sbatch sbatch_train_clf.sh non 0.0 only_supervised 2000

sbatch sbatch_train_clf.sh sym 0.1 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.2 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.3 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.4 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.5 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.6 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.7 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.8 only_supervised 2000
sbatch sbatch_train_clf.sh sym 0.9 only_supervised 2000

sbatch sbatch_train_clf.sh asym 0.1 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.2 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.3 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.4 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.5 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.6 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.7 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.8 only_supervised 2000
sbatch sbatch_train_clf.sh asym 0.9 only_supervised 2000

