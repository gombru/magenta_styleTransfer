from magenta.models.image_stylization import image_stylization_transform
from PIL import Image
import os
import warnings
import random

warnings.filterwarnings("ignore")

results_path = "/home/Imatge/ssd2/ICDAR_2015_IndidentalSceneText/train/img_styled_icdar/"

num_styles = 96 # 9, 32, 34
# checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/multistyle-pastiche-generator-varied.ckpt"
# checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/miro"
checkpoint = "/home/Imatge/hd/datasets/styleTransferMiro/train/icdar"

# which_styles = []
# for i in range(num_styles): which_styles.append(i)

input_images_dir = "/home/Imatge/ssd2/ICDAR_2015_IndidentalSceneText/train/img/"
input_images = []
for file in os.listdir(input_images_dir): input_images.append(file.split('/')[-1])
# legible_ids_dir = "/home/Imatge/ssd2/COCO-Text/gt_COCO_format_legible/"
# legible_ids = []
# for file in os.listdir(legible_ids_dir): legible_ids.append(file.split('/')[-1].strip('.json'))

# final_ids = [id for id in input_images if id.strip('.jpg') in legible_ids]
# del input_images
# del legible_ids

# print("Number of images with legible text: " + str(len(final_ids)))

batch_size = 32
i=0

while True:
    cur_styles = random.sample(range(0, 96), 4)
    # cur_styles.remove(0)
    # cur_styles.remove(6)
    # cur_styles.remove(25)
    print(" --> Starting batch from" + str(i) + " with styles " + str(cur_styles))

    if i > len(input_images):
        break

    last_image = i + batch_size
    if last_image > len(input_images):
        last_image = len(input_images)

    cur_input_images = input_images[i:last_image]

    result_images = image_stylization_transform.multiple_input_images(checkpoint, num_styles, input_images_dir, cur_input_images, cur_styles)

    for k, v in result_images.items():
        v = v[0,:,:,:]
        pil_image = Image.fromarray((v*255).astype('uint8'))
        pil_image.save(results_path + k + '.png')

    i+=batch_size
    print(" --> " + str(i) + " out of " + str(len(input_images)))


print("DONE")


