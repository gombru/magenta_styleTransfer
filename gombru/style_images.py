from magenta.models.image_stylization import image_stylization_transform
import Image
import os

results_path = "/home/raulgomez/datasets/StyleTransfer/results/full_words/"

num_styles = 34 # 9, 32, 34
# checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/multistyle-pastiche-generator-varied.ckpt"
# checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/miro"
checkpoint = "/home/raulgomez/other_datasets/styleTransferMiro/models/words/"

which_styles = []
for i in range(num_styles): which_styles.append(i)

input_images_dir = "/home/raulgomez/datasets/COCO-Text/img/img_minival_2/"
input_images = []
for file in os.listdir(input_images_dir): input_images.append(file.split('/')[-1])

# input_images = input_images[15:]

print("Computing ...")
result_images = image_stylization_transform.multiple_input_images(checkpoint, num_styles, input_images_dir, input_images, which_styles)

# Save result images
print("Saving results")
for k, v in result_images.iteritems():
    v = v[0,:,:,:]
    pil_image = Image.fromarray((v*255).astype('uint8'))
    pil_image.save(results_path + k + '.png')

print("DONE")


