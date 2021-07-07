## Studying the behaviour of SSL objectives in presence of noise 

## Environment setup
```
virtualenv --system-site-packages -p python3 env_ssl
source env_ssl/bin/activate
pip install -r requirements.txt
```

We train contrastive and non-contrastive SSL models (with lightly) using unlabeled CIFAR-10 training images; follwed by training an image classifier model using noisy CIFAR-10 data (with different levels of symmetric and assymetric noise). We evaluate the classifier on (noise-free) CIFAR-10 test set.

### Train MoCo 
'''
python moco/train_moco.py --max-epochs 1000
'''

### Train Classifier
'''
python moco/train_classifier.py --max-epochs 100
'''
