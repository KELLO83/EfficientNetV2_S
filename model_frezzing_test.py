import timm
import torch
import torchinfo

model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True, num_classes=2)

torchinfo.summary(
    model=model,
    input_size=(1, 3, 320, 320),
    verbose=2,)
    


# model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=2)

print('='*50)

# torchinfo.summary(
#     model=model,
#     input_size=(1, 3, 320, 320),
#     verbose=2,

for param in model.parameters():
    param.requires_grad = False

print("Unfreezing parameters in model.blocks[5]...")
for param in model.blocks[5][14].parameters() :
    param.requires_grad = True

# for name , param in model.named_parameters():
#     if name.startswith('blocks.5.13') or name.startswith('blocks.5.14'):
#         param.requires_grad = True


print("Unfreezing parameters in model.classifier...")
for param in model.classifier.parameters():
    param.requires_grad = True

for parm in model.conv_head.parameters():
    parm.requires_grad = True

print('='*50)

print("Verifying parameter freeze status:")
total_params = 0
trainable_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"- {name} (Trainable)")

print('='*50)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")



input("모든 파라미터 확인 추가작업 아무키..")

for name , param in model.named_parameters():
    print(f"name : {name} , requires_grad : {param.requires_grad}")