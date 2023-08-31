import os
from PIL import Image

folders = ['10240','10289','10291','10345','10387']
#folders = ['10406','10444','10526','10005','10017','10028','10059']

for folder in folders:
    

    og_image_dir = "/home/pathway/mydisk/Project/Protein_data/" + folder + "/changed/images"
    og_masks_dir = "/home/pathway/mydisk/Project/Protein_data/" + folder + "/changed/masks"

    re_folder = "/home/pathway/mydisk/Project/Protein_data/" + folder + "/resized"

    re_image_dir = os.path.join(re_folder, 'images')
    re_masks_dir = os.path.join(re_folder, 'masks')
    os.makedirs(re_image_dir, exist_ok=True)
    os.makedirs(re_masks_dir, exist_ok=True)

    t = len(os.listdir(og_image_dir))
    im = 0
    
    for filename in os.listdir(og_image_dir):
        im = im+1;
        image_path = os.path.join(og_image_dir, filename)
        image = Image.open(image_path)

        re_image = image.resize((256, 256))

        output_path = os.path.join(re_image_dir, filename)
        re_image.save(output_path)
        print(folder," --> image -->",im,"/",t)
        image.close()

    ma = 0
    for filename in os.listdir(og_masks_dir):
        ma = ma + 1;
        masks_path = os.path.join(og_masks_dir, filename)
        image = Image.open(masks_path)

        re_masks = image.resize((256, 256))

        output_path = os.path.join(re_masks_dir, filename)
        re_masks.save(output_path)
        print(folder," --> masks -->",ma,"/",t)
        image.close()
