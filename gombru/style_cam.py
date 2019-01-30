from magenta.models.image_stylization import image_stylization_transform
import styleTranfer_calls

checkpoint = "/home/raulgomez/datasets/styleTransferMiro/models/miro/"
num_styles=9
which_style=[8]
SaveVideo = False

styleTranfer_calls.style_from_camera(checkpoint, num_styles, which_style, SaveVideo)
