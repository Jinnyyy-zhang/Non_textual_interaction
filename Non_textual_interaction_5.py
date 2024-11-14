import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import re
import torch
from tqdm import tqdm

# 设置设备：如果有GPU则使用GPU
device = 0 if torch.cuda.is_available() else -1

# 加载情感分类模型
tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, device=device)

# 加载意图检测模型
intention_detector = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# 加载特征提取模型
text_similarity = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2", device=device)

# 情感和意图标签映射
emotion_mapping = {
    "happiness": 1, "joy": 1, "amusement": 1, "approval": 1,
    "love": 2, "caring": 2, "admiration": 2,
    "anger": 3, "annoyance": 3, "disapproval": 3,
    "sadness": 4, "grief": 4, "remorse": 4, "confusion": 4,
    "fear": 5, "nervousness": 5,
    "disgust": 6, "contempt": 6,
    "surprise": 7, "realization": 7,
}

intention_mapping = {
    "entertaining": 3,
    "expressive": 2,
    "offensive": 4,
    "interactive": 1,
    "other": 5
}


# 辅助函数：提取标签中的数字部分
def extract_number_from_label(label):
    match = re.match(r"(\d+)", label)
    return int(match.group(1)) if match else None

# 修改E_text.csv中的text列内容并保存为E_text2.csv
def modify_and_save_text(text_filepath, label_filepath, output_filepath):
    text_df = pd.read_csv(text_filepath, encoding="ISO-8859-1")
    label_df = pd.read_csv(label_filepath, encoding="ISO-8859-1")
    
    # 只在创建 E_text2.csv 时进行一次性追加
    for index, row in label_df.iterrows():
        if pd.notna(row['metaphor category']):
            source = row.get('source domain', '')
            target = row.get('target domain', '')
            addition = f"{source} in the picture means {target}."
            file_name = row['file_name']

            text_df.loc[text_df['file_name'] == file_name, 'text'] += ' ' + addition
            
    
    text_df.to_csv(output_filepath, index=False, encoding="ISO-8859-1")


# 调用修改函数
modify_and_save_text("meme_en/E_text.csv", "meme_en/label_E.csv", "meme_en/E_text2.csv")

# 加载数据集
def load_dataset(csv_path, image_folder_path, text_filepath="meme_en/E_text2.csv"):
    with open(csv_path, 'r', encoding="ISO-8859-1") as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    
    text_data = pd.read_csv(text_filepath, encoding="ISO-8859-1")
    text_dict = dict(zip(text_data['file_name'], text_data['text']))
    
    for row in data:
        row.append(os.path.join(image_folder_path, row[0]))  
        meme_text = text_dict.get(row[0], "")
        row.append(meme_text)
        
        # 提取纯数字部分，更新情感类别和意图标签为整数
        row[1] = extract_number_from_label(row[1].strip())  # 将情感类别转换为纯数字
        row[3] = extract_number_from_label(row[3].strip())  # 将意图标签转换为纯数字

       
    return data

# 辅助函数：提取数值等级
def extract_level(text):
    if isinstance(text, (float, int)):
        return int(text)
    if isinstance(text, str):
        match = re.match(r'(\d+)', text)
        return int(match.group()) if match else 0
    return 0

# 聊天内容分析，获取情绪、意图、特征
def process_chat_content(chat_text):
    # 获取情绪类别
    emotion_scores = classifier(chat_text)[0]
    goemotions_label = max(emotion_scores, key=lambda x: x['score'])['label']
    sentiment_category = emotion_mapping.get(goemotions_label, None)
    sentiment_degree = max(emotion_scores, key=lambda x: x['score'])['score']
    
    # 使用零样本分类模型进行意图检测
    candidate_labels = list(intention_mapping.keys())
    intention_result = intention_detector(chat_text, candidate_labels=candidate_labels)
    detected_intention = intention_result["labels"][0] if intention_result["labels"] else "other"
    
    # 映射到意图类别，若未找到则设置为默认值5 (other)
    intention = intention_mapping.get(detected_intention, 5)  
    
    # 获取情感特征
    chat_features = np.array(text_similarity(chat_text)[0])
    
    return sentiment_category, sentiment_degree, intention, chat_features.reshape(1, -1)

# 匹配表情包并计算相似度得分
def match_meme(chat_features, meme_database, chat_sentiment_category, chat_sentiment_degree, chat_intention):
    matched_memes = []
    
    for row in meme_database:
        file_name = row[0]
        meme_text = row[-1]
        meme_sentiment_category = row[1]  # 已提取为纯数字
        meme_sentiment_degree = extract_level(row[2])
        meme_intention_level = row[3]  # 已提取为纯数字
        
        # 提取聊天情感强度和意图等级
        chat_sentiment_degree_level = extract_level(chat_sentiment_degree)
        chat_intention_level = extract_level(chat_intention) if chat_intention is not None else 5
        
        if not isinstance(meme_text, str) or meme_text.strip() == "":
            continue
        
        # 计算特征相似度
        meme_features = np.array(text_similarity(meme_text)[0]).reshape(1, -1)
        
        if chat_features.shape[1] != meme_features.shape[1]:
            min_dim = min(chat_features.shape[1], meme_features.shape[1])
            chat_features_matched = chat_features[:, :min_dim]
            meme_features_matched = meme_features[:, :min_dim]
            feature_similarity = cosine_similarity(chat_features_matched, meme_features_matched)[0][0]
        else:
            feature_similarity = cosine_similarity(chat_features, meme_features)[0][0]
        
        # 计算情感类别相似度
        sentiment_category_similarity = 1.0 if chat_sentiment_category == meme_sentiment_category else 0.0
        
        # 计算情感强度和意图相似度
        sentiment_degree_difference = abs(chat_sentiment_degree_level - meme_sentiment_degree)
        sentiment_degree_similarity = 1 - (sentiment_degree_difference / 4)
        intention_difference = abs(chat_intention_level - meme_intention_level)
        intention_similarity = 1 - (intention_difference / 4)
        
        # 综合相似度
        total_similarity = (
            0.2 * feature_similarity +
            0.4 * sentiment_category_similarity +
            0.3 * sentiment_degree_similarity +
            0.1 * intention_similarity
        )
        
        matched_memes.append((file_name, total_similarity, feature_similarity, sentiment_category_similarity, sentiment_degree_similarity, intention_similarity))
    
    matched_memes.sort(key=lambda x: x[1], reverse=True)
    return matched_memes[:5]

# 主函数
def main(chat_text, csv_path, image_folder_path, text_filepath):
    meme_database = load_dataset(csv_path, image_folder_path, text_filepath)
    sentiment_category, sentiment_degree, intention, chat_features = process_chat_content(chat_text)
    
    recommendations = match_meme(
        chat_features, meme_database, 
        chat_sentiment_category=sentiment_category, 
        chat_sentiment_degree=sentiment_degree, 
        chat_intention=intention
    )
    
    return {
        "情感类别": sentiment_category,
        "情感强度": f"{sentiment_degree}",
        "检测意图": intention,
        "推荐表情包": [
            {
                "文件名": f"{name}",
                "综合相似度": f"{score:.2%}",
                "特征相似度": f"{feature_similarity:.2%}",
                "情感类别相似度": f"{sentiment_category_similarity:.2%}",
                "情感强度相似度": f"{sentiment_degree_similarity:.2%}",
                "意图相似度": f"{intention_similarity:.2%}",
                "图片路径": os.path.join(image_folder_path, f"{name}")
            } for sub_id, (name, score, feature_similarity, sentiment_category_similarity, sentiment_degree_similarity, intention_similarity) in enumerate(recommendations)
        ]
    }

# 示例q_list和路径设置
q_list = [
    "I am very happy today! And I want eat some delicious food!", 
    "I’m so tired today. All I want to do is go home, make some tea, and relax.",
    "I’m in a great mood! Maybe I’ll treat myself to something nice at the mall.",
    "Feeling a bit down. I think I’ll just curl up on the couch and binge-watch something.",
    "The weather’s perfect! I should take my bike out and go for a ride around the park.",
    "I’ve been procrastinating so much—today, I need to check off a few things from my list!",
    "I’m kinda bored. Maybe I’ll text a friend I haven’t talked to in a while and catch up.",
    "Feeling awesome today! I definitely deserve a good meal tonight!",
    "Today feels so chill. It might be a good time to go through some old stuff I’ve kept around.",
    "Suddenly, I’m in the mood to learn something new. Maybe there’s a cool online class I can take!",
    "I’m feeling a little anxious… A walk outside might help clear my head.",
    "I’m in a weird mood. Maybe I’ll just put on some music and zone out for a bit.",
    "I really want to talk to someone! I feel like I’ve got so much to share lately.",
    "Today, I’m feeling super motivated! I should get to the gym and get a good workout in.",
    "I miss my family a bit. Maybe I’ll give them a call later and see how everyone’s doing.",
    "I’ve been feeling kinda lost… Maybe writing down my thoughts will help me figure things out.",
    "The weather’s so nice! I’m thinking about heading to the beach for a little walk.",
    "I’m a bit frustrated today. Maybe doing something with my hands, like crafting, will help.",
    "The sun is shining! Perfect excuse to grab a book and go sit in a cozy café nearby.",
    "I’m not feeling my best… I think some alone time would do me good." 
]

csv_path = "meme_en/label_E.csv"
image_folder_path = "meme_en/Eimages"
text_filepath = "meme_en/E_text2.csv"

# 主函数调用和结果保存
with open("metadata_new.csv", "w", newline='') as new_file, open("metadata2.csv", "w", newline='') as meta_file:
    new_writer = csv.writer(new_file)
    meta_writer = csv.writer(meta_file)
    
    new_writer.writerow(["file_name", "id", "sub_id", "question"])
    meta_writer.writerow(["file_name", "id", "sub_id", "question", "情感类别", "情感强度", "检测意图", "推荐表情包", "综合相似度", "特征相似度", "情感类别相似度", "情感强度相似度", "意图相似度"])
    
    # 使用 tqdm 显示进度条
    for j, q in enumerate(tqdm(q_list, desc="Processing questions")):
        result = main(
            q, 
            csv_path=csv_path, 
            image_folder_path=image_folder_path, 
            text_filepath=text_filepath
        )
        
        for i, img_dict in enumerate(result["推荐表情包"]):
            new_file_name = img_dict["文件名"]  
            
            new_writer.writerow([new_file_name, j, f'{j}_{i}', q])
            
            meta_writer.writerow([
                new_file_name, j, f'{j}_{i}', q, result["情感类别"], result["情感强度"], result["检测意图"], img_dict["文件名"],
                img_dict["综合相似度"], img_dict["特征相似度"], img_dict["情感类别相似度"], img_dict["情感强度相似度"], img_dict["意图相似度"]
            ])
