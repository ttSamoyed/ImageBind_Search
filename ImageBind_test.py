import data
import torch
import faiss
from models import imagebind_model
from models.imagebind_model import ModalityType


# 所有变量
text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

total_list = text_list+image_paths+audio_paths

print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 初始化model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

select_mode = input("请选择模式：1.文本输入 2.图片输入 3.音频输入")
input_file = input("请输入目标文本（若为图片、音频请输入文件路径）")

inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

if int(select_mode)==1:
    print("正在处理文本...")
    target = {
        ModalityType.TEXT: data.load_and_transform_text([input_file], device),
    }
elif int(select_mode)==2:
    print("正在处理图片...")
    target = {
        ModalityType.VISION: data.load_and_transform_vision_data([input_file], device),
    }
else:
    print("正在处理声音...")
    target = {
        ModalityType.AUDIO: data.load_and_transform_audio_data([input_file], device),
    }


with torch.no_grad():
    embeddings_inputs = model(inputs)
    embeddings_target = model(target)

print("文字embedding：", embeddings_inputs[ModalityType.TEXT])
print("图片embedding：", embeddings_inputs[ModalityType.VISION])
print("声音embedding：", embeddings_inputs[ModalityType.AUDIO])
d = embeddings_inputs[ModalityType.VISION].size(1)
index = faiss.IndexFlatL2(d)
index.add(embeddings_inputs[ModalityType.TEXT].cpu().numpy())
index.add(embeddings_inputs[ModalityType.VISION].cpu().numpy())
index.add(embeddings_inputs[ModalityType.AUDIO].cpu().numpy())
print(f"当前向量总数为：{index.ntotal}")

if int(select_mode)==1:
    print("正在构建输入文本embedding...")
    input_file_embedding = embeddings_target[ModalityType.TEXT]
elif int(select_mode)==2:
    print("正在构建输入图片embedding...")
    input_file_embedding = embeddings_target[ModalityType.VISION]
else:
    print("正在构建输入音频embedding...")
    input_file_embedding = embeddings_target[ModalityType.AUDIO]

print("上传文件embedding:", input_file_embedding)


k = 9
D , I = index.search(input_file_embedding.cpu().numpy(),k)
print(D)
print(I)

for i in range(k):
    print(f"多模态搜索的第{i}匹配结果为：{total_list[I[0][i]]}")