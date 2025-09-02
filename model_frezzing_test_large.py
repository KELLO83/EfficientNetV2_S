
import timm

model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True, num_classes=2)
print(model)
input("모델 확인 후 아무키...")
for param in model.parameters():
    param.requires_grad = False


for param in model.blocks[6][6].parameters() :
    param.requires_grad = True

for parm in model.blocks[6][5].parameters() :
    parm.requires_grad = True

print("Unfreezing parameters in model.classifier...")
for param in model.classifier.parameters():
    param.requires_grad = True

for parm in model.conv_head.parameters():
    parm.requires_grad = True

for parm in model.bn2.parameters():
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