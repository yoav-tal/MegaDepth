
import torch
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from models.models import create_model
model = create_model(opt)

model.switch_to_eval()







def get_mask(img, model=model):

    img = np.float32(img)/255.0

    input_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()
    input_img = input_img.unsqueeze(0)

    #input_images = Variable(input_img.cpu())

    pred_log_depth = model.netG.forward(input_img)
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)


    #log_prediction = log_prediction.squeeze(0)
    #prediction = torch.exp(log_prediction[0,:,:])

    pred_inv_depth = pred_depth#1 / pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()

    pred_inv_depth = pred_inv_depth / np.amax(pred_inv_depth)

    return pred_inv_depth


