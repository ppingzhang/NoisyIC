from torchvision.models import vgg19, vgg16
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models
import os



class VGG19_LossNetwork(torch.nn.Module):
    def __init__(self):
        super(VGG19_LossNetwork, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vgg_model = vgg19(pretrained=False).features[:].to(device) 
        vgg_model.load_state_dict(torch.load('./loss/vgg19-dcbb9e9d.pth'),strict=False)
        vgg_model.eval()
         
        for param in vgg_model.parameters():
            param.requires_grad = False 


        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '31': "relu5_2"
        }
        #self.weight = [1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]
        self.weight =[1.0,1.0,1.0,1.0,1.0]
    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            #print("vgg_layers name:",name,module)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #print(output.keys())
        return list(output.values())
 
    def forward(self, output, gt):
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter,(dehaze_feature, gt_feature,loss_weight) in enumerate(zip(output_features, gt_features,self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature)*loss_weight)
        return sum(loss), output_features  #/len(loss)


if __name__ == '__main__':
    import numpy as np
    vgg = VGG19_LossNetwork()
    output = torch.tensor(np.zeros([1, 3, 64, 64]), dtype=torch.float)
    gt = torch.tensor(np.ones([1, 3, 64, 64]), dtype=torch.float)
    loss, feats = vgg(output, gt)
    print(loss)

