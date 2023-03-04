# Dataset

- Home page: https://www.aitex.es/afid/
- Cropped version: 



# Train

- ```shell
  python train.py --config ./config.yml \
  				--root_dir ./ \
  				--train_dir ./datasets/AITEX/Crop/train \
  				--checkpoint_dir ./ckpts \
  				--train_anomaly \
  				--test_dir ./datasets/AITEX/Crop/test
  ```

  

# Test

- ```shell
  python train.py --config ./config.yml \
  				--root_dir ./ \
  				--anomaly \
  				--test_dir ./datasets/AITEX/Crop/test \
  				--test_result_dir ./sample/test_result \
  				--pretrained \
  				--checkpoint_dir ./ckpts
  ```

  

