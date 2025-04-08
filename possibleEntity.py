import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def discover_dict_categories(raw_data, n_samples=None):
    """
    从原始地理位置数据中自动发现词典类别
    """
    # 如果指定了样本数，随机抽样
    if n_samples and n_samples < len(raw_data):
        np.random.seed(42)
        samples = np.random.choice(raw_data, n_samples, replace=False)
    else:
        samples = raw_data

    # 提取文本内容
    texts = []
    for entry in tqdm(samples, desc="提取文本"):
        if ' with entities "' in entry:
            text = entry.strip().split(' with entities "')[0].strip('"')
            texts.append(text)
        else:
            texts.append(entry.strip())

    # 1. 使用NLP工具进行实体识别
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    entity_types = Counter()
    unique_entities = {}

    for text in tqdm(texts, desc="NLP分析"):
        doc = nlp(text)
        for ent in doc.ents:
            entity_types[ent.label_] += 1
            if ent.label_ not in unique_entities:
                unique_entities[ent.label_] = set()
            unique_entities[ent.label_].add(ent.text)

    print("\n自动识别的实体类型:")
    for entity_type, count in entity_types.most_common():
        print(f"{entity_type}: {count} 个实例")

    # 2. 使用正则表达式提取特定模式
    patterns = {
        "可能的水体": r'([A-Z][a-z]+(?: [A-Z][a-z]+)* (?:River|Lake|Bayou|Creek|Pond|Reservoir|Springs?))',
        "可能的方向": r'\b([NSEW]\.?|[NS][EW]\.?|North|South|East|West)\b',
        "可能的距离": r'(\d+\.?\d*)\s*(?:mi(?:le)?|km)s?',
        "可能的相对位置": r'\b(above|below|near|at|mouth of|junction)\b',
        "可能的坐标": r'([TRS]\d+[NSEW]|Sec\.\s*\d+)',
        "可能的行政区域": r'([A-Z][a-z]+ (?:County|Parish|Township))',
        "可能的道路": r'((?:Hwy|Highway|Rt|Route|US|CR|FM)\s*\d+)'
    }

    regex_entities = {}
    for pattern_name, pattern in patterns.items():
        regex_entities[pattern_name] = set()
        for text in texts:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # 如果正则返回捕获组
                regex_entities[pattern_name].add(match)

    print("\n基于模式识别的可能实体类型:")
    for pattern_name, entities in regex_entities.items():
        print(f"{pattern_name}: {len(entities)} 个唯一实例")
        print(f"示例: {list(entities)[:5]}")

    # 3. 使用n-gram和聚类发现未知模式
    vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=5)
    X = vectorizer.fit_transform(texts)

    # 将文档向量转换为距离矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(X)
    distance_matrix = 1 - similarity

    # 处理负值问题 - 保留负值处理，但确保DBSCAN和TSNE可以正常工作
    # 检查是否有负值
    if np.any(distance_matrix < 0):
        print(f"警告：距离矩阵中包含 {np.sum(distance_matrix < 0)} 个负值，最小值为 {np.min(distance_matrix)}")
        print("这些可能是由数值精度问题引起的。将对DBSCAN和TSNE使用经过处理的非负距离矩阵。")

        # 创建一个非负版本用于DBSCAN和TSNE
        distance_matrix_non_negative = np.maximum(0, distance_matrix)
    else:
        distance_matrix_non_negative = distance_matrix

    # 使用DBSCAN聚类
    clustering = DBSCAN(eps=0.5, min_samples=3, metric='precomputed').fit(distance_matrix_non_negative)

    # 分析聚类结果
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"\n自动发现了 {n_clusters} 个潜在地理特征聚类")

    # 分析每个聚类的特征词
    cluster_features = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # 跳过噪声

        # 获取该聚类的文档
        cluster_docs = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]

        # 创建一个新的向量化器，只考虑这个聚类的文档
        cluster_vectorizer = CountVectorizer(ngram_range=(1, 2))
        cluster_X = cluster_vectorizer.fit_transform(cluster_docs)

        # 计算平均TF-IDF权重
        feature_names = cluster_vectorizer.get_feature_names_out()
        cluster_features[cluster_id] = []

        # 保存该聚类的前20个特征词
        cluster_sum = cluster_X.sum(axis=0)
        words_freq = [(word, cluster_sum[0, idx]) for word, idx in
                      zip(feature_names, range(len(feature_names)))]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        cluster_features[cluster_id] = words_freq[:20]

    print("\n每个聚类的典型特征词:")
    for cluster_id, features in cluster_features.items():
        print(f"聚类 {cluster_id}:")
        print(", ".join([f"{word} ({freq})" for word, freq in features[:10]]))
        print(f"示例文本: {texts[list(labels).index(cluster_id)]}")
        print("-" * 50)

    # 4. 基于上下文相似性发现词典类别
    # 将单词转换为向量表示
    from gensim.models import Word2Vec

    # 准备数据
    tokenized_texts = [text.lower().split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

    # 寻找相似单词
    word_clusters = {}
    seed_words = ['river', 'lake', 'bayou', 'highway', 'north', 'south', 'mile', 'county']

    for seed in seed_words:
        if seed in model.wv:
            similar_words = model.wv.most_similar(seed, topn=20)
            word_clusters[seed] = similar_words

    print("\n基于词向量相似度的潜在词典类别:")
    for seed, similar in word_clusters.items():
        print(f"与 '{seed}' 相似的词:")
        print(", ".join([f"{word} ({round(score, 2)})" for word, score in similar]))

    # 5. 绘制聚类可视化 - 使用非负距离矩阵进行TSNE
    if n_clusters > 0:
        from sklearn.manifold import TSNE

        # 使用X.toarray()来获取稠密矩阵，确保没有负值
        X_array = X.toarray()
        X_non_negative = np.abs(X_array)  # 使用绝对值来确保非负

        tsne = TSNE(n_components=2, random_state=42)
        try:
            X_reduced = tsne.fit_transform(X_non_negative)

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster ID')
            plt.title('t-SNE visualization of geographic location clusters')
            plt.savefig('geo_clusters.png')
            plt.close()
            print("\nt-SNE可视化已保存到 'geo_clusters.png'")
        except Exception as e:
            print(f"\nt-SNE可视化失败: {e}")
            print("这可能是由于维度降维过程中的数值问题引起的。")

    # 返回发现的实体类型和示例
    return {
        "nlp_entities": {k: list(v) for k, v in unique_entities.items()},
        "regex_entities": {k: list(v) for k, v in regex_entities.items()},
        "clusters": cluster_features,
        "word_vectors": word_clusters
    }


def analyze_discovered_categories(discoveries):
    """深入分析发现的词典类别"""
    all_categories = set()

    # 合并所有发现的类别
    for source in ['nlp_entities', 'regex_entities']:
        for category in discoveries[source].keys():
            all_categories.add(category)

    # 分析聚类是否揭示了新的类别
    for cluster_id, features in discoveries['clusters'].items():
        # 提取关键词
        keywords = [word for word, _ in features[:5]]
        keywords_str = " ".join(keywords)

        # 基于关键词猜测可能的类别
        if any(word in keywords_str.lower() for word in ['river', 'lake', 'bayou', 'creek']):
            print(f"聚类 {cluster_id} 可能是另一种水体类型")
        elif any(word in keywords_str.lower() for word in ['highway', 'road', 'route', 'hwy']):
            print(f"聚类 {cluster_id} 可能是道路类型")
        elif any(word in keywords_str.lower() for word in ['north', 'south', 'east', 'west', 'direction']):
            print(f"聚类 {cluster_id} 可能是方向类型")
        elif any(word in keywords_str.lower() for word in ['mile', 'km', 'meter', 'distance']):
            print(f"聚类 {cluster_id} 可能是距离类型")
        elif any(word in keywords_str.lower() for word in ['county', 'parish', 'state', 'city', 'town']):
            print(f"聚类 {cluster_id} 可能是行政区域类型")
        else:
            print(f"聚类 {cluster_id} 可能是新的词典类型！关键词: {keywords}")

    # 返回所有发现的类别
    return all_categories

# 从文件读取原始数据
with open('geo_locations.txt', 'r') as f:
    raw_data = f.readlines()

# 分析数据发现词典类型（可以限制样本数量以加快处理速度）
discoveries = discover_dict_categories(raw_data, n_samples=1000)

# 分析发现的类别
all_categories = analyze_discovered_categories(discoveries)
print(f"\n总共发现了 {len(all_categories)} 个潜在词典类别")