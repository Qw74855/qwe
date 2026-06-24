import csv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 配置路径
csv_path = "./data/campus_data.csv"
output_chunk_path = "./data/text_chunks.txt"

# 校验CSV文件是否存在
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"文件不存在：{csv_path}，请检查文件路径")

# 1. 读取csv拼接问答文本
all_text = ""
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row["question"].strip()
        a = row["answer"].strip()
        all_text += f"问题：{q} 回答：{a}\n\n"  # 增加空行分隔，优化分割优先级

# 2. 分割器优化：自定义分割符，优先按问答换行切割/
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 自定义分割优先级
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)

# 3. 执行切分
chunks = text_splitter.split_text(all_text)

# 4. 打印统计信息
print(f"原始总文本字符数：{len(all_text)}")
print(f"切分片段总数：{len(chunks)}")
print("-" * 50)

# 打印并保存分段结果
with open(output_chunk_path, "w", encoding="utf-8") as out_f:
    for idx, chunk in enumerate(chunks):
        info = f"【分段{idx+1} | 字符长度：{len(chunk)}】\n{chunk}\n" + "-"*30 + "\n"
        print(info)
        out_f.write(info)

print(f"\n所有分段已保存至：{output_chunk_path}")