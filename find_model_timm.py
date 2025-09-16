

import timm

model_list = timm.list_models('convnext*_laion*', pretrained=True)
model_list = timm.list_models('convnext*', pretrained=True)
model_list = timm.list_models('convnextv2_*fcmae*', pretrained=True)
for m in model_list:
    print(m)


# model  = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in1k', pretrained=True , num_classes=2)

# import torchinfo
# print(model)
# torchinfo.summary(model, input_size=(1, 3, 320 , 320))


# efficient_list = timm.list_models('tf_efficientnetv2*.in21k_ft_in1k*', pretrained=True)


# for m in efficient_list:
#     print(m)

    

