import Image
import os

results_path = "/home/raulgomez/datasets/StyleTransfer/results/averaged_styles/"
new_results_path = "/home/raulgomez/datasets/StyleTransfer/results/averaged_styles_processed/"

for file in os.listdir(results_path):
    image = Image.open(results_path + file)
    image.save(new_results_path + file)

print("DONE")
