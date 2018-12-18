from magenta.models.image_stylization import image_stylization_transform
import Image

results_path = "/home/raulgomez/datasets/styleTransferMiro/results_processed/"

num_styles=32
checkpoint="/home/raulgomez/datasets/styleTransferMiro/models/multistyle-pastiche-generator-varied.ckpt"
which_styles=[0,1,2,5]

input_images_dir = "/home/raulgomez/datasets/styleTransferMiro/test_images/"
input_images = ["me.jpg","me2.jpg"]

print("Computing ...")
result_images = image_stylization_transform.multiple_input_images(checkpoint, num_styles, input_images_dir, input_images, which_styles)

# Save result images
print("Saving results")
for k, v in result_images.iteritems():
    v = v[0,:,:,:]
    pil_image = Image.fromarray((v*255).astype('uint8'))
    pil_image.save(results_path + k + '.png')

print("DONE")


