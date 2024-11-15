import csv
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 加载描述数据和图片路径
data = pd.read_csv("output_with_images.csv")  # 包含 'text' 和 'image_path' 列

# 初始化零样本分类器
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def recommend_emojis(user_input, top_n=5):
    """
    根据用户输入推荐表情包。
    """
    # 提取描述和图片路径
    descriptions = data['text'].tolist()
    image_paths = data['image_path'].tolist()
    
    # 使用零样本分类器预测最相关的表情包
    results = classifier(user_input, descriptions)
    
    # 获取前 top_n 个最相关的表情描述及其索引
    recommended_indices = []
    seen_descriptions = set()
    
    for i, label in enumerate(results['labels']):
        if label not in seen_descriptions:  # 避免重复推荐
            seen_descriptions.add(label)
            index = descriptions.index(label)
            recommended_indices.append(index)
        if len(recommended_indices) >= top_n:
            break
    
    # 返回推荐的图片路径和描述
    return [(image_paths[i], descriptions[i]) for i in recommended_indices]

# 示例用户输入
user_inputs = [
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

# 输出结果 CSV 文件
output_file = "final_recommendations.csv"

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name", "id", "sub_id", "question"])  # 写入标题行
    
    for question_id, user_input in enumerate(tqdm(user_inputs, desc="Processing user inputs"), start=1):
        recommendations = recommend_emojis(user_input, top_n=5)
        
        # 写入每个推荐的表情包
        for sub_id, (file_name, description) in enumerate(recommendations, start=1):
            writer.writerow([file_name, question_id, f"{question_id}_{sub_id}", user_input])

print(f"Recommendations have been saved to {output_file}.")
