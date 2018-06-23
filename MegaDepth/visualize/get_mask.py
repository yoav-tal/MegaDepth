
import torch
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

from models.models import create_model
model = create_model(opt)

model.switch_to_eval()

#input = torch.zeros([1,3,16,16])

#output = model.netG.forward(input)





def get_mask(img, model=model):

    img = np.float32(img)/255.0
    img = np.expand_dims(img, axis=0)

    #torch_img = torch.from_numpy(np.transpose(img, (2,0,1)) ).contiguous().float()
    torch_img = torch.from_numpy(img)
    torch_img = torch_img.permute(0,3,1,2)
    torch_img = torch_img.float()
    #torch_img = torch_img.unsqueeze(0)

    log_prediction = model.netG.forward(torch_img)
    log_prediction = log_prediction.squeeze(0)
    prediction = torch.exp(log_prediction[0,:,:])

    inv_depth_pred = 1/prediction
    inv_depth_pred = inv_depth_pred.data.numpy()

    return inv_depth_pred/np.amax(inv_depth_pred)


