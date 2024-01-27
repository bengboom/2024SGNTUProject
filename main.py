import numpy as np
import os
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import urllib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import ViTImageProcessor, ViTForImageClassification

# 用户提供的图像URL
user_image_path = "https://dr.ntu.edu.sg/cris/rp/fileservice/rp00631/57/?filename=ttteoh_1_2.JPG"

# 加载模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
# vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# 加载和处理图像的函数
def load_and_process_image(image_path_or_url):
    if urllib.parse.urlparse(image_path_or_url).scheme in ('http', 'https'):
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path_or_url)
    img = img.convert("RGB")
    return img

# 提取特征的函数
def extract_features_resnet(image_path):
    img = load_and_process_image(image_path)
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input_resnet(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

def extract_features_vit(image_path):
    img = load_and_process_image(image_path)
    img = vit_processor(images=img, return_tensors="np").pixel_values
    img = img.flatten()
    return img

# 计算余弦相似度的函数
def calculate_cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm1 = np.linalg.norm(feature1)
    norm2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

# 文件夹路径，包含所有角色的图像
character_folder = "./genshin_jpgs"
character_features_resnet = {}
character_features_vit = {}

# 为每个角色图像提取特征
for character_image in os.listdir(character_folder):
    if character_image.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(character_folder, character_image)
        features_resnet = extract_features_resnet(image_path)
        features_vit = extract_features_vit(image_path)
        character_name = character_image.split('.')[0]
        character_features_resnet[character_name] = features_resnet
        character_features_vit[character_name] = features_vit

# 找到最相似的角色
def find_most_similar_character(features, character_features):
    max_similarity = 0
    most_similar_character = None

    for character, feature in character_features.items():
        similarity = calculate_cosine_similarity(features, feature)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_character = character

    return most_similar_character, max_similarity

user_features_resnet = extract_features_resnet(user_image_path)
user_features_vit = extract_features_vit(user_image_path)

most_similar_character_resnet, similarity_resnet = find_most_similar_character(user_features_resnet, character_features_resnet)
most_similar_character_vit, similarity_vit = find_most_similar_character(user_features_vit, character_features_vit)


# 加载和处理图片
user_photo = load_and_process_image(user_image_path)
resnet_photo = Image.open(os.path.join(character_folder, most_similar_character_resnet + ".jpeg"))
vit_photo = Image.open(os.path.join(character_folder, most_similar_character_vit + ".jpeg"))

# 添加标题和相似度
#user_photo = add_title_and_similarity_to_image(user_photo, "User Photo", 0)  # 用户照片的相似度设为0
#resnet_photo = add_title_and_similarity_to_image(resnet_photo, "ResNet Similar Character", similarity_resnet)
#vit_photo = add_title_and_similarity_to_image(vit_photo, "ViT Similar Character", similarity_vit)

# 合并图片的函数
def merge_images(images, spacing=10):
    total_width = sum(img.size[0] for img in images) + spacing * (len(images) - 1)
    max_height = max(img.size[1] for img in images)
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        new_im.paste(img, (x_offset, 0))
        x_offset += img.size[0] + spacing

    return new_im

# 合并图片并保存
merged_image = merge_images([user_photo, resnet_photo, vit_photo])
merged_image.save("output.jpg")

print("Resnet similarity", similarity_resnet)
print("VIT similarity", similarity_vit)