import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import json
from tqdm import tqdm
import spacy
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns


class GeoGazetteerBuilder:
    """地理位置词典构建器"""

    def __init__(self, min_occurrence=2, similarity_threshold=0.7):
        """
        初始化词典构建器

        参数:
        - min_occurrence: 最小出现次数，用于过滤低频项
        - similarity_threshold: 相似度阈值，用于聚类相似项
        """
        self.min_occurrence = min_occurrence
        self.similarity_threshold = similarity_threshold
        self.data = []
        self.processed_data = []
        self.word_vectors = None
        self.gazetteers = {}
        self.patterns = {}
        self.hierarchical_gazetteer = {}
        self.synonyms = {}

        # 预先定义主要类别
        self.primary_categories = [
            "water_bodies", "directions", "distances",
            "relative_positions", "admin_areas", "roads",
            "coordinates", "geographical_features"
        ]

        # 尝试加载spaCy模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def load_data(self, file_path=None, raw_data=None, n_samples=None):
        """
        加载原始数据

        参数:
        - file_path: 数据文件路径
        - raw_data: 直接提供的原始数据
        - n_samples: 随机抽样数量
        """
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = f.readlines()
        elif raw_data:
            self.data = raw_data
        else:
            raise ValueError("需要提供file_path或raw_data")

        # 随机抽样
        if n_samples and n_samples < len(self.data):
            np.random.seed(42)
            self.data = list(np.random.choice(self.data, n_samples, replace=False))

        # 提取文本内容
        self.texts = []
        for entry in tqdm(self.data, desc="提取文本"):
            if ' with entities "' in entry:
                text = entry.strip().split(' with entities "')[0].strip('"')
                self.texts.append(text)
            else:
                self.texts.append(entry.strip())

        print(f"加载了 {len(self.texts)} 条地理位置描述")
        return self

    def preprocess_data(self):
        """预处理数据，提取结构化信息"""
        self.processed_data = []

        # 定义模式
        patterns = {
            "water_bodies": r'([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)* (?:River|Lake|Bayou|Creek|Springs?|Reservoir|Canal|Pond|Bay|Sound|Gulf|Stream|Brook|Fork|Waters?|Inlet|Cove|Basin|Strait|Channel)s?)',
            "directions": r'\b([NSEW]\.?|[NS][EW]\.?|North|South|East|West|Northern|Southern|Eastern|Western)\b',
            "distances": r'(\d+\.?\d*)\s*(?:mi(?:le)?|km|m|ft|feet|yard)s?',
            "relative_positions": r'\b(above|below|near|at|mouth of|junction|entrance|upstream|downstream|across|along|between|opposite|adjacent|inside|outside|behind|front|center)\b',
            "coordinates": r'([TRS]\d+[NSEW]|Sec\.\s*\d+|\d{1,2}[°\-]\d{1,2}[\'′\-](?:\d{1,2})?[\"″]?\s*[NSEW]|Lat|Long)',
            "admin_areas": r'([A-Z][a-z]+ (?:County|Parish|Township|City|Town|Village|State)|[A-Z]{2})',
            "roads": r'((?:Hwy|Highway|Rt|Route|US|CR|FM|Interstate|I-|St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard|Bridge)\s*\d+[A-Za-z]?)',
            "geographical_features": r'\b(Island|Swamp|Marsh|Mountain|Valley|Hill|Ridge|Prairie|Beach|Shore|Bank|Falls|Spring|Basin|Peninsula|Cape|Point|Head|Bend)\b'
        }

        for text in tqdm(self.texts, desc="结构化处理"):
            data_item = {
                "original_text": text,
            }

            # 使用模式提取实体
            for category, pattern in patterns.items():
                matches = re.findall(pattern, text)
                if isinstance(matches, list) and matches and isinstance(matches[0], tuple):
                    matches = [m[0] for m in matches]  # 处理捕获组
                data_item[category] = list(set(matches))

            # 使用spaCy提取实体
            doc = self.nlp(text)
            spacy_entities = defaultdict(list)
            for ent in doc.ents:
                spacy_entities[ent.label_].append(ent.text)

            data_item["spacy_entities"] = dict(spacy_entities)

            # 添加到处理后的数据
            self.processed_data.append(data_item)

        print(f"完成 {len(self.processed_data)} 条数据的结构化处理")
        return self

    def build_word_vectors(self):
        """构建词向量模型"""
        # 准备数据
        tokenized_texts = [text.lower().split() for text in self.texts]

        # 训练词向量模型
        self.word_vectors = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=self.min_occurrence,
            workers=4
        )

        print(f"词向量模型包含 {len(self.word_vectors.wv.key_to_index)} 个词语")
        return self

    def build_primary_gazetteers(self):
        """构建主要词典"""
        # 初始化所有主要类别的词典
        for category in self.primary_categories:
            self.gazetteers[category] = Counter()

        # 收集所有类别的词条
        for item in tqdm(self.processed_data, desc="构建主要词典"):
            for category in self.primary_categories:
                if category in item and item[category]:
                    for entry in item[category]:
                        self.gazetteers[category][entry] += 1

        # 过滤低频项
        for category in self.primary_categories:
            self.gazetteers[category] = {
                k: v for k, v in self.gazetteers[category].items()
                if v >= self.min_occurrence
            }

        # 打印统计信息
        for category, entries in self.gazetteers.items():
            print(f"{category}: {len(entries)} 个唯一项")

        return self

    def build_secondary_gazetteers(self):
        """构建二级词典"""
        # 水体子类型
        water_types = ["River", "Lake", "Bayou", "Creek", "Pond", "Reservoir", "Gulf", "Bay", "Stream", "Brook"]
        self.hierarchical_gazetteer["water_bodies"] = {}

        for water_type in water_types:
            self.hierarchical_gazetteer["water_bodies"][water_type.lower()] = [
                entry for entry in self.gazetteers["water_bodies"]
                if f" {water_type}" in entry or entry.endswith(water_type)
            ]

        # 方向子类型
        direction_types = {
            "cardinal": ["North", "South", "East", "West", "N", "S", "E", "W"],
            "composite": ["Northeast", "Northwest", "Southeast", "Southwest", "NE", "NW", "SE", "SW"],
            "relative": ["above", "below", "upstream", "downstream", "left", "right"]
        }

        self.hierarchical_gazetteer["directions"] = {}
        for subtype, keywords in direction_types.items():
            self.hierarchical_gazetteer["directions"][subtype] = []
            for keyword in keywords:
                self.hierarchical_gazetteer["directions"][subtype].extend([
                    entry for entry in self.gazetteers["directions"]
                    if keyword.lower() in entry.lower() or keyword.upper() in entry.upper()
                ])

        # 相对位置子类型
        position_types = {
            "proximity": ["near", "close", "adjacent", "by", "at"],
            "relation": ["above", "below", "upstream", "downstream", "between"],
            "boundary": ["mouth", "entrance", "junction", "confluence", "source"]
        }

        self.hierarchical_gazetteer["relative_positions"] = {}
        for subtype, keywords in position_types.items():
            self.hierarchical_gazetteer["relative_positions"][subtype] = []
            for keyword in keywords:
                self.hierarchical_gazetteer["relative_positions"][subtype].extend([
                    entry for entry in self.gazetteers["relative_positions"]
                    if keyword.lower() in entry.lower()
                ])

        # 打印统计信息
        for category, subtypes in self.hierarchical_gazetteer.items():
            print(f"\n{category} 子类型:")
            for subtype, entries in subtypes.items():
                print(f"  {subtype}: {len(set(entries))} 个唯一项")

        return self

    def build_synonym_dictionary(self):
        """构建同义词词典"""
        # 初始化同义词词典
        self.synonyms = {}

        # 方向同义词
        self.synonyms["directions"] = {
            "north": ["N", "N.", "North", "Northern"],
            "south": ["S", "S.", "South", "Southern"],
            "east": ["E", "E.", "East", "Eastern"],
            "west": ["W", "W.", "West", "Western"],
            "northeast": ["NE", "NE.", "Northeast", "Northeastern"],
            "northwest": ["NW", "NW.", "Northwest", "Northwestern"],
            "southeast": ["SE", "SE.", "Southeast", "Southeastern"],
            "southwest": ["SW", "SW.", "Southwest", "Southwestern"]
        }

        # 距离同义词
        self.synonyms["distances"] = {
            "mile": ["mi", "mi.", "mile", "miles"],
            "kilometer": ["km", "km.", "kilometer", "kilometers"],
            "meter": ["m", "m.", "meter", "meters"],
            "foot": ["ft", "ft.", "foot", "feet"]
        }

        # 相对位置同义词
        self.synonyms["relative_positions"] = {
            "near": ["near", "close to", "by", "adjacent to", "next to"],
            "at": ["at", "on", "in", "along", "within"],
            "above": ["above", "upstream of", "upstream from", "north of"],
            "below": ["below", "downstream of", "downstream from", "south of"],
            "mouth": ["mouth of", "outlet of", "entrance to", "exit of"],
            "junction": ["junction", "confluence", "intersection", "meeting"]
        }

        # 道路同义词
        self.synonyms["roads"] = {
            "highway": ["Hwy", "Hwy.", "Highway"],
            "route": ["Rt", "Rt.", "Route"],
            "road": ["Rd", "Rd.", "Road"],
            "street": ["St", "St.", "Street"],
            "avenue": ["Ave", "Ave.", "Avenue"],
            "boulevard": ["Blvd", "Blvd.", "Boulevard"]
        }

        # 使用词向量扩展同义词
        if self.word_vectors:
            for seed_word in ["river", "lake", "bayou", "north", "south", "mile", "county"]:
                if seed_word in self.word_vectors.wv:
                    similar_words = self.word_vectors.wv.most_similar(seed_word, topn=5)
                    if seed_word not in self.synonyms:
                        category = self._get_category_for_word(seed_word)
                        if category:
                            if category not in self.synonyms:
                                self.synonyms[category] = {}
                            self.synonyms[category][seed_word] = [w for w, s in similar_words if
                                                                  s > self.similarity_threshold]

        # 打印同义词统计
        for category, synonym_groups in self.synonyms.items():
            print(f"\n{category} 同义词组:")
            for key_word, synonyms in synonym_groups.items():
                print(f"  {key_word}: {', '.join(synonyms)}")

        return self

    def _get_category_for_word(self, word):
        """
        确定单词所属的类别
        """
        category_keywords = {
            "water_bodies": ["river", "lake", "bayou", "creek", "pond", "reservoir", "gulf"],
            "directions": ["north", "south", "east", "west", "northeast", "northwest"],
            "distances": ["mile", "km", "meter", "foot", "yard", "distance"],
            "relative_positions": ["near", "at", "above", "below", "mouth", "junction"],
            "admin_areas": ["county", "parish", "township", "city", "town", "state"],
            "roads": ["highway", "route", "road", "street", "avenue", "boulevard"],
            "geographical_features": ["island", "swamp", "marsh", "mountain", "valley", "hill"]
        }

        for category, keywords in category_keywords.items():
            if word.lower() in keywords:
                return category

        return None

    def build_pattern_dictionary(self):
        """构建模式词典"""
        # 定义各种类型的模式
        self.patterns = {
            # 水体描述模式
            "water_body_patterns": [
                r'([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)* (?:River|Lake|Bayou|Creek))',
                r'([A-Z][a-zA-Z\'\-]+ (?:River|Lake|Bayou|Creek)) at',
                r'([A-Z][a-zA-Z\'\-]+ (?:River|Lake|Bayou|Creek)) near'
            ],

            # 位置描述模式
            "location_patterns": [
                r'at ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)',
                r'near ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)',
                r'((?:\d+\.?\d*)\s*(?:mi(?:le)?|km)s?) (?:[NSEW]\.?|[NS][EW]\.?) of ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)',
                r'mouth of ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)'
            ],

            # 方向和距离组合模式
            "direction_distance_patterns": [
                r'((?:\d+\.?\d*)\s*(?:mi(?:le)?|km)s?) ([NSEW]\.?|[NS][EW]\.?)',
                r'([NSEW]\.?|[NS][EW]\.?) of',
                r'([NSEW]\.?|[NS][EW]\.?) side',
                r'([NSEW]\.?|[NS][EW]\.?) (?:from|to)'
            ],

            # 坐标模式
            "coordinate_patterns": [
                r'([TRS]\d+[NSEW]|Sec\.\s*\d+)',
                r'(\d{1,2}[°\-]\d{1,2}[\'′\-](?:\d{1,2})?[\"″]?\s*[NSEW])',
                r'Lat\.?\s*(\d{1,2}[°\-]\d{1,2}[\'′\-](?:\d{1,2})?[\"″]?)',
                r'Long\.?\s*(\d{1,2}[°\-]\d{1,2}[\'′\-](?:\d{1,2})?[\"″]?)'
            ],

            # 复合地理描述模式
            "complex_geo_patterns": [
                r'([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)* (?:River|Lake|Bayou|Creek)) (?:at|near) ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)',
                r'([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*) (?:County|Parish) (?:line|border)',
                r'bridge (?:on|at|over) ([A-Z][a-zA-Z\'\-]+(?: [A-Z][a-zA-Z\'\-]+)*)',
                r'((?:Hwy|Highway|Rt|Route|US|CR|FM|Interstate|I-|St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard)\s*\d+[A-Za-z]?) (?:crossing|bridge)'
            ]
        }

        # 统计每个模式在数据中的匹配次数
        pattern_matches = {}

        for category, pattern_list in self.patterns.items():
            pattern_matches[category] = {}

            for i, pattern in enumerate(pattern_list):
                pattern_matches[category][f"pattern_{i + 1}"] = 0

                for text in self.texts:
                    matches = re.findall(pattern, text)
                    if matches:
                        pattern_matches[category][f"pattern_{i + 1}"] += len(matches)

        # 打印模式匹配统计
        for category, patterns in pattern_matches.items():
            print(f"\n{category} 模式匹配统计:")
            for pattern_id, count in patterns.items():
                print(f"  {pattern_id}: 匹配 {count} 次")

        return self

    def build_gazetteer(self):
        """
        构建完整的词典
        """
        # 执行所有构建步骤
        return (self
                .preprocess_data()
                .build_word_vectors()
                .build_primary_gazetteers()
                .build_secondary_gazetteers()
                .build_synonym_dictionary()
                .build_pattern_dictionary())

    def save_gazetteers(self, output_dir="gazetteers"):
        """
        保存所有词典到文件
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存主要词典
        for category, entries in self.gazetteers.items():
            with open(os.path.join(output_dir, f"{category}.txt"), "w", encoding="utf-8") as f:
                for entry, count in sorted(entries.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{entry}\t{count}\n")

        # 保存层次词典
        for category, subtypes in self.hierarchical_gazetteer.items():
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            for subtype, entries in subtypes.items():
                with open(os.path.join(category_dir, f"{subtype}.txt"), "w", encoding="utf-8") as f:
                    for entry in sorted(set(entries)):
                        f.write(f"{entry}\n")

        # 保存同义词词典
        with open(os.path.join(output_dir, "synonyms.json"), "w", encoding="utf-8") as f:
            json.dump(self.synonyms, f, indent=2)

        # 保存模式词典
        with open(os.path.join(output_dir, "patterns.json"), "w", encoding="utf-8") as f:
            json.dump(self.patterns, f, indent=2)

        # 保存词向量模型
        if self.word_vectors:
            self.word_vectors.save(os.path.join(output_dir, "word_vectors.model"))

        # 创建词典索引文件
        index_content = {
            "primary_categories": self.primary_categories,
            "statistics": {
                category: len(entries) for category, entries in self.gazetteers.items()
            },
            "secondary_categories": {
                category: list(subtypes.keys())
                for category, subtypes in self.hierarchical_gazetteer.items()
            },
            "synonym_categories": list(self.synonyms.keys()),
            "pattern_categories": list(self.patterns.keys())
        }

        with open(os.path.join(output_dir, "gazetteer_index.json"), "w", encoding="utf-8") as f:
            json.dump(index_content, f, indent=2)

        print(f"所有词典已保存到 {output_dir} 目录")
        return self

    def visualize_gazetteer_stats(self, output_dir="gazetteers"):
        """
        可视化词典统计信息
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 各类别词条数量
        category_counts = {
            category: len(entries) for category, entries in self.gazetteers.items()
        }

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
        plt.title("各类别词条数量")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "category_counts.png"))
        plt.close()

        # 水体子类型分布
        if "water_bodies" in self.hierarchical_gazetteer:
            water_subtype_counts = {
                subtype: len(entries)
                for subtype, entries in self.hierarchical_gazetteer["water_bodies"].items()
            }

            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(water_subtype_counts.keys()), y=list(water_subtype_counts.values()))
            plt.title("水体子类型分布")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "water_subtypes.png"))
            plt.close()

        if self.word_vectors:
            # Select key words
            key_words = ["river", "lake", "bayou", "creek", "north", "south", "mile", "county"]
            key_word_vectors = {}

            for word in key_words:
                if word in self.word_vectors.wv:
                    key_word_vectors[word] = self.word_vectors.wv[word]

            if len(key_word_vectors) > 1:
                # Use t-SNE for dimensionality reduction
                from sklearn.manifold import TSNE

                vectors = np.array(list(key_word_vectors.values()))
                n_samples = len(vectors)

                # Set perplexity to a value lower than the number of samples
                perplexity = min(30, n_samples - 1)

                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced_vectors = tsne.fit_transform(vectors)

                # 可视化
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

                for i, word in enumerate(key_word_vectors.keys()):
                    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

                plt.title("关键词向量空间分布")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "word_vectors.png"))
                plt.close()

        print(f"统计可视化已保存到 {output_dir} 目录")
        return self


# 示例用法
def main():
    # 从文件读取原始数据
    file_path = "geo_locations.txt"

    # 创建词典构建器并执行构建流程
    gazetteer_builder = GeoGazetteerBuilder(min_occurrence=2)

    # 加载数据 (可以选择限制样本数量以加快处理)
    gazetteer_builder.load_data(file_path, n_samples=None)

    # 构建和保存词典
    gazetteer_builder.build_gazetteer().save_gazetteers()

    # 可视化词典统计
    gazetteer_builder.visualize_gazetteer_stats()

    print("词典构建完成!")


if __name__ == "__main__":
    main()