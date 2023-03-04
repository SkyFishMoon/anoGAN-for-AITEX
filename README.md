# Dataset

- Home page: https://www.aitex.es/afid/

- Cropped version: https://drive.google.com/file/d/1iDBS92WVHR50GLga77cn0RrleC6rrlEs/view?usp=sharing

  Download and unzip it to the `datasets/AITEX/` directory

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


# Reference

- https://github.com/seokinj/anoGAN
