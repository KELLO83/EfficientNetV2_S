

import timm

model_list = timm.list_models('convnext*_laion*', pretrained=True)
# for m in model_list:
#     print(m)







model  = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in1k', pretrained=True , num_classes=2)

import torchinfo
print(model)
torchinfo.summary(model, input_size=(1, 3, 320 , 320))