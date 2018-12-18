from magenta.models.image_stylization import image_stylization_transform
import styleTranfer_calls

checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/multistyle-pastiche-generator-varied.ckpt"
num_styles=32
which_style=[5]
SaveVideo = False

styleTranfer_calls.style_from_camera(checkpoint, num_styles, which_style, SaveVideo)
