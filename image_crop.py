from PIL import Image
import os
original_nodefect_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/NoDefect_all_images'
original_defect_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/Defect_images'
original_defect_mask_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/Mask_images'
crop_defect_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/Crop/defect'
crop_nodefect_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/Crop/nodefect'
crop_defect_mask_path = '/home/tianyu/Desktop/work_space/anoGAN/anoGAN/datasets/AITEX/Crop/mask'

for img_name in os.listdir(original_nodefect_path):
    img = Image.open(original_nodefect_path + '/' + img_name)
    for i in range(16):
        left = 0 + i * 256
        right = 256 + i * 256
        crop = img.crop((left, 0, right, 256))
        crop.save(crop_nodefect_path + '/' + img_name[:-4] + '-' + str(i) + '.png')

for img_name in os.listdir(original_defect_path):
    img = Image.open(original_defect_path + '/' + img_name)
    for i in range(16):
        left = 0 + i * 256
        right = 256 + i * 256
        crop = img.crop((left, 0, right, 256))
        crop.save(crop_defect_path + '/' + img_name[:-4] + '-' + str(i) + '.png')

for img_name in os.listdir(original_defect_mask_path):
    img = Image.open(original_defect_mask_path + '/' + img_name)
    for i in range(16):
        left = 0 + i * 256
        right = 256 + i * 256
        crop = img.crop((left, 0, right, 256))
        crop.save(crop_defect_mask_path + '/' + img_name[:-4] + '-' + str(i) + '.png')

