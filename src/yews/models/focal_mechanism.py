import torch.nn as nn

from .utils import load_state_dict_from_url

# import torch
# try:
#     from torch.hub import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
__all__ = ['FmV1', 'fm_v1', 'FmV2', 'fm_v2']

model_urls = {
    'fm_v1': 'https://www.dropbox.com/s/ckb4glf35agi9xa/fm_v1_wenchuan-bdd92da2.pth?dl=1',
    'fm_v2': 'https://www.dropbox.com/s/ckb4glf35agi9xa/fm_v1_wenchuan-bdd92da2.pth?dl=1',
}

class FmV1(nn.Module):

    def __init__(self):
        super().__init__()
        # n*3*71*9000
        # nn.Cov2d(3,16,kernel_size=(3,7),stride=1,padding=....)
        # (3,7) 3>71 7>18000 in the end use (3,3).

        # 71,9000 -> 71,4507
        self.layer1 = nn.Linear(1704*9000,10)
        self.layer2 = nn.Linear(10,5)
        self.layer3 = nn.Linear(5,3)
     
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out)

        return out

def fm_v1(pretrained=False, progress=True, **kwargs):
    r"""Original CPIC model architecture from the
    `"Deep learning for ..." <https://arxiv.org/abs/1901.06396>`_ paper. The
    pretrained model is trained on 60,000 Wenchuan aftershock dataset
    demonstrated in the paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Wenchuan)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = FmV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fm_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class FmV2(nn.Module):
    
    #https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    
    def __init__(self):
        super(FmV2, self).__init__()
        self.features = nn.Sequential(

            # 71,9000 -> 71,4507
            
            nn.Conv2d(3, 16, kernel_size=(3,17), stride=1, padding=(2,16), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),
            
            # 71,4507 -> 71,2260

            nn.Conv2d(16, 32, kernel_size=(3,15), stride=1, padding=(2,14), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),

            # 71,2260 -> 71,1135

            nn.Conv2d(32, 64, kernel_size=(3,13), stride=1, padding=(2,12), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),


            # 71,1135 -> 71,572

            nn.Conv2d(64, 64, kernel_size=(3,11), stride=1, padding=(2,10), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),


            # 71,572 -> 71,289

            nn.Conv2d(64, 64, kernel_size=(3,9), stride=1, padding=(2,8), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),


            # 71,289 -> 71,143

            nn.Conv2d(64, 64, kernel_size=(3,7), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),


            # 71,143 -> 71,71

            nn.Conv2d(64, 64, kernel_size=(3,5), stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
#             nn.Dropout(0.1),

            # 71,71 -> 35,35

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
#             nn.Dropout(0.1),


            # 35,35 -> 17,17

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
#             nn.Dropout(0.1),

        
            # 17,17 -> 8,8

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
#             nn.Dropout(0.1),
                        
            # 8,8 -> 4,4

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
#             nn.Dropout(0.1),
                        
#             # 4,4 -> 2,2

#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
# #             nn.Dropout(0.1),
                        
#             # 2,2 -> 1,1

#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
# #             nn.Dropout(0.1),
                        

        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1),
        )
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

def fm_v2(pretrained=False, progress=True, **kwargs):

    model = FmV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['fm_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

# if __name__ == '__main__':
#     model = fm_v1(pretrained=False)
    
#     x = torch.ones([1, 3, 71, 9000])
#     out = model(x)
#     print(out.size())
