# 地理位置同义词库构建和实体识别模型训练
import random

import pandas as pd
import numpy as np
import re
import string
import json
from collections import defaultdict
import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 确保下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')


# 1. 数据加载和预处理
def load_and_preprocess_data(file_path):
    """
    加载并预处理locality数据
    """
    print("加载并预处理数据...")

    # 加载数据
    # 如果是CSV文件
    df = pd.read_csv(file_path,delimiter='\t')
    # 如果是Excel文件
    # df = pd.read_excel(file_path)

    # 检查并重命名列名以匹配示例数据
    expected_columns = [
        'LocalityString', 'Drainage', 'Country',
        'State', 'County', 'Continent'
    ]

    # 确保所有列都存在
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None

    # 清理数据
    df = df.fillna("[NULL]")

    # 处理文本数据
    df['LocalityString'] = df['LocalityString'].astype(str).apply(
        lambda x: x.strip() if x != "[NULL]" else x
    )

    print(f"加载了 {len(df)} 条记录")
    return df


# 2. 同义词库构建
class GeoSynonymBuilder:
    def __init__(self):
        # 初始化同义词类别
        self.synonym_network = {
            "water_bodies": defaultdict(list),
            "geographic_features": defaultdict(list),
            "directional_terms": defaultdict(list),
            "relative_positions": defaultdict(list),
            "administrative_areas": defaultdict(list),
            "distance_units": defaultdict(list),
            "abbreviations": defaultdict(list)
        }

        # 预填充常见同义词
        self._initialize_synonym_network()

    def _initialize_synonym_network(self):
        """
        预填充基本的地理同义词
        """
        # 水体同义词
        self.synonym_network["water_bodies"] = {
            "river": ["stream", "creek", "brook", "tributary", "trib.", "rivulet", "branch", "run", "bayou", "fork"],
            "lake": ["pond", "reservoir", "lagoon", "bay", "backwater", "pool"],
            "ocean": ["sea", "gulf", "strait"]
        }

        # 地理特征
        self.synonym_network["geographic_features"] = dict(mountain=["mt", "mt.", "mount", "peak", "ridge", "hill"],
                                                           island=["isle", "isl.", "islet", "key", "cay"],
                                                           point=["pt", "pt.", "peninsula", "promontory", "headland",
                                                                  "cape"])

        # 方向词
        self.synonym_network["directional_terms"] = {
            "north": ["n", "n.", "northern"],
            "south": ["s", "s.", "southern"],
            "east": ["e", "e.", "eastern"],
            "west": ["w", "w.", "western"],
            "northeast": ["ne", "n.e.", "northeastern"],
            "northwest": ["nw", "n.w.", "northwestern"],
            "southeast": ["se", "s.e.", "southeastern"],
            "southwest": ["sw", "s.w.", "southwestern"]
        }

        # 相对位置
        self.synonym_network["relative_positions"] = {
            "above": ["upstream", "upriver", "upstream from", "higher than"],
            "below": ["downstream", "downriver", "downstream from", "lower than"],
            "mouth": ["confluence", "outlet", "junction"],
            "tributary": ["trib.", "tributary of", "branch of", "affluent"]
        }

        # 行政区域
        self.synonym_network["administrative_areas"] = {
            "county": ["co.", "co", "parish"],
            "township": ["twp", "twp.", "t", "town"],
            "range": ["r", "r."],
            "section": ["sec", "sec.", "s", "s."]
        }

        # 距离单位
        self.synonym_network["distance_units"] = {
            "mile": ["mi", "mi.", "miles"],
            "kilometer": ["km", "kilometers"],
            "meter": ["m", "meters"],
            "foot": ["ft", "ft.", "feet"]
        }

        # 州名缩写
        self.synonym_network["abbreviations"] = {
            "alabama": ["al", "ala"],
            "alaska": ["ak"],
            "arizona": ["az", "ariz"],
            "arkansas": ["ar", "ark"],
            "california": ["ca", "calif", "cali"],
            "colorado": ["co", "colo"],
            "connecticut": ["ct", "conn"],
            "delaware": ["de", "del"],
            "florida": ["fl", "fla"],
            "georgia": ["ga"],
            "hawaii": ["hi"],
            "idaho": ["id"],
            "illinois": ["il", "ill"],
            "indiana": ["in", "ind"],
            "iowa": ["ia"],
            "kansas": ["ks", "kan"],
            "kentucky": ["ky", "ken"],
            "louisiana": ["la"],
            "maine": ["me"],
            "maryland": ["md"],
            "massachusetts": ["ma", "mass"],
            "michigan": ["mi", "mich"],
            "minnesota": ["mn", "minn"],
            "mississippi": ["ms", "miss"],
            "missouri": ["mo"],
            "montana": ["mt", "mont"],
            "nebraska": ["ne", "neb", "nebr"],
            "nevada": ["nv", "nev"],
            "new hampshire": ["nh"],
            "new jersey": ["nj"],
            "new mexico": ["nm"],
            "new york": ["ny"],
            "north carolina": ["nc"],
            "north dakota": ["nd"],
            "ohio": ["oh"],
            "oklahoma": ["ok", "okla"],
            "oregon": ["or", "ore"],
            "pennsylvania": ["pa", "penn"],
            "rhode island": ["ri"],
            "south carolina": ["sc"],
            "south dakota": ["sd"],
            "tennessee": ["tn", "tenn"],
            "texas": ["tx", "tex"],
            "utah": ["ut"],
            "vermont": ["vt"],
            "virginia": ["va", "virg"],
            "washington": ["wa", "wash"],
            "west virginia": ["wv", "w va", "w. va"],
            "wisconsin": ["wi", "wis", "wisc"],
            "wyoming": ["wy", "wyo"]
        }

    def extract_potential_synonyms(self, df):
        """
        从数据中提取潜在的同义词
        """
        print("从数据中提取潜在同义词...")

        # 提取所有unique的locality值
        localities = df['LocalityString'].unique().tolist()

        # 1. 提取水体名称
        water_bodies = self._extract_water_bodies(localities)

        # 2. 提取相对位置描述
        relative_positions = self._extract_relative_positions(localities)

        # 3. 提取特有地名和可能的变体
        place_names = self._extract_place_names(localities)

        # 4. 寻找拼写变体
        spelling_variants = self._find_spelling_variants(place_names)

        # 5. 分析共现模式
        cooccurrence_patterns = self._analyze_cooccurrence(localities)

        # 整合提取的同义词
        self._integrate_extracted_synonyms(
            water_bodies,
            relative_positions,
            spelling_variants,
            cooccurrence_patterns
        )

        print("潜在同义词提取完成")
        return self.synonym_network

    def _extract_water_bodies(self, localities):
        """
        提取水体名称
        """
        water_bodies = []

        # 水体相关词汇
        water_terms = ["River", "Creek", "Lake", "Bay", "Sea", "Ocean",
                       "Pond", "Stream", "Brook", "Trib", "Pool", "Backwater"]

        pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s(?:' + '|'.join(water_terms) + r')'

        for locality in localities:
            matches = re.findall(pattern, locality)
            for match in matches:
                water_bodies.append(match)

        return list(set(water_bodies))

    def _extract_relative_positions(self, localities):
        """
        提取相对位置描述
        """
        positions = []

        # 相对位置词汇
        position_terms = [
            "above", "below", "upstream", "downstream",
            "mouth", "confluence", "junction", "crossing",
            "tributary", "trib"
        ]

        for locality in localities:
            for term in position_terms:
                if term in locality.lower():
                    context = re.findall(r'\w+\s' + term + r'\s\w+', locality.lower())
                    positions.extend(context)

        return list(set(positions))

    def _extract_place_names(self, localities):
        """
        提取地名
        """
        place_names = []

        # 地名通常是首字母大写的词组
        pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)'

        for locality in localities:
            matches = re.findall(pattern, locality)
            for match in matches:
                # 过滤掉常见的非地名词汇
                if not any(term in match for term in ["River", "Creek", "Lake"]):
                    place_names.append(match)

        return list(set(place_names))

    def _find_spelling_variants(self, terms):
        """
        寻找拼写变体
        """
        variants = {}

        # 对每一对词计算Levenshtein距离
        for i, term1 in enumerate(terms):
            for j, term2 in enumerate(terms):
                if i >= j:
                    continue

                # 计算相似度
                distance = Levenshtein.distance(term1.lower(), term2.lower())
                similarity = 1 - (distance / max(len(term1), len(term2)))

                # 如果相似度高但不完全相同，可能是拼写变体
                if similarity > 0.8 and similarity < 1.0:
                    if term1 not in variants:
                        variants[term1] = []
                    variants[term1].append(term2)

        return variants

    def _analyze_cooccurrence(self, localities):
        """
        分析词汇共现模式
        """
        # 创建文档-词矩阵
        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]+\b',
            stop_words='english'
        )

        # 转换为矩阵
        X = vectorizer.fit_transform(localities)

        # 计算词共现矩阵
        word_to_idx = vectorizer.vocabulary_
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        # 计算共现 (X^T * X)
        co_occurrence_matrix = (X.T * X).toarray()

        # 寻找高度共现的词对
        patterns = {}

        threshold = 0.5  # 共现阈值
        for i in range(len(idx_to_word)):
            for j in range(i + 1, len(idx_to_word)):
                co_occurrence = co_occurrence_matrix[i, j]

                # 计算共现强度
                word_i_count = co_occurrence_matrix[i, i]
                word_j_count = co_occurrence_matrix[j, j]

                if word_i_count > 0 and word_j_count > 0:
                    strength = co_occurrence / (word_i_count + word_j_count - co_occurrence)

                    if strength > threshold:
                        word_i = idx_to_word[i]
                        word_j = idx_to_word[j]

                        if word_i not in patterns:
                            patterns[word_i] = []
                        patterns[word_i].append((word_j, strength))

        return patterns

    def _integrate_extracted_synonyms(self, water_bodies, relative_positions,
                                      spelling_variants, cooccurrence_patterns):
        """
        整合提取的同义词到网络中
        """
        # 整合水体变体
        for water_body in water_bodies:
            water_type = self._determine_water_body_type(water_body)
            if water_type:
                self.synonym_network["water_bodies"][water_type].append(water_body)

        # 整合相对位置
        for position in relative_positions:
            position_type = self._determine_position_type(position)
            if position_type:
                self.synonym_network["relative_positions"][position_type].append(position)

        # 整合拼写变体
        for primary, variants in spelling_variants.items():
            # 尝试确定变体类型
            category, type_key = self._determine_term_category(primary)
            if category and type_key:
                self.synonym_network[category][type_key].extend(variants)

        # 整合共现模式
        for word, co_words in cooccurrence_patterns.items():
            category, type_key = self._determine_term_category(word)
            if category and type_key:
                for co_word, strength in co_words:
                    if co_word not in self.synonym_network[category][type_key]:
                        self.synonym_network[category][type_key].append(co_word)

    def _determine_water_body_type(self, term):
        """
        确定水体类型
        """
        term_lower = term.lower()
        if any(river_term in term_lower for river_term in ["river", "creek", "stream", "brook"]):
            return "river"
        elif any(lake_term in term_lower for lake_term in ["lake", "pond", "reservoir"]):
            return "lake"
        elif any(ocean_term in term_lower for ocean_term in ["ocean", "sea", "gulf"]):
            return "ocean"
        return None

    def _determine_position_type(self, term):
        """
        确定位置描述类型
        """
        term_lower = term.lower()
        if any(above_term in term_lower for above_term in ["above", "upstream"]):
            return "above"
        elif any(below_term in term_lower for below_term in ["below", "downstream"]):
            return "below"
        elif any(mouth_term in term_lower for mouth_term in ["mouth", "confluence"]):
            return "mouth"
        elif any(trib_term in term_lower for trib_term in ["tributary", "trib"]):
            return "tributary"
        return None

    def _determine_term_category(self, term):
        """
        确定术语类别
        """
        term_lower = term.lower()

        # 检查水体
        if any(water_term in term_lower for water_term in ["river", "creek", "lake", "pond"]):
            return "water_bodies", self._determine_water_body_type(term)

        # 检查方向
        if any(dir_term in term_lower for dir_term in ["north", "south", "east", "west"]):
            for key, values in self.synonym_network["directional_terms"].items():
                if term_lower in [key] + [v.lower() for v in values]:
                    return "directional_terms", key

        # 检查相对位置
        if any(pos_term in term_lower for pos_term in ["above", "below", "mouth", "tributary"]):
            return "relative_positions", self._determine_position_type(term)

        # 检查行政区域
        if any(admin_term in term_lower for admin_term in ["county", "township", "range"]):
            for key, values in self.synonym_network["administrative_areas"].items():
                if term_lower in [key] + [v.lower() for v in values]:
                    return "administrative_areas", key

        # 检查距离单位
        if any(dist_term in term_lower for dist_term in ["mile", "kilometer", "meter"]):
            for key, values in self.synonym_network["distance_units"].items():
                if term_lower in [key] + [v.lower() for v in values]:
                    return "distance_units", key

        # 检查州名缩写
        for state_name, abbrevs in self.synonym_network["abbreviations"].items():
            if term_lower == state_name or term_lower in [a.lower() for a in abbrevs]:
                return "abbreviations", state_name

        return None, None

    def export_to_elasticsearch_format(self):
        """
        将同义词网络导出为Elasticsearch格式
        """
        es_synonyms = []

        for category, category_dict in self.synonym_network.items():
            for primary_term, variants in category_dict.items():
                if variants:
                    # 移除重复项
                    unique_variants = list(set([v.lower() for v in variants if v.lower() != primary_term.lower()]))
                    if unique_variants:
                        synonym_line = f"{primary_term}, {', '.join(unique_variants)}"
                        es_synonyms.append(synonym_line)

        return es_synonyms

    def save_to_file(self, file_path):
        """
        保存同义词网络到文件
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.synonym_network, f, indent=2)

        print(f"同义词网络已保存到 {file_path}")


# 3. 训练地理实体识别模型
class GeoNERTrainer:
    def __init__(self):
        self.labels = [
            "WATER_BODY", "ADMIN_AREA", "DIRECTION",
            "DISTANCE", "LANDMARK", "COORDINATE", "RELATIVE_POSITION"
        ]
        self.model = None

    def create_training_data(self, df):
        """
        创建训练数据
        """
        print("创建NER训练数据...")

        # 使用规则生成初步标注
        training_data = []

        for idx, row in df.iterrows():
            locality = row['LocalityString']
            if isinstance(locality, str) and locality != "[NULL]":
                # 使用规则识别实体
                entities = self._rule_based_entity_recognition(locality)

                # 添加到训练数据
                if entities:
                    training_data.append((locality, {"entities": entities}))

        print(f"创建了 {len(training_data)} 条训练数据")
        return training_data

    def _rule_based_entity_recognition(self, text):
        """
        基于规则的实体识别，避免实体重叠
        """
        candidate_entities = []

        # 水体识别规则
        water_body_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s(River|Creek|Stream|Brook|Lake|Pond|Bay|Sea)'
        for match in re.finditer(water_body_pattern, text):
            start, end = match.span()
            candidate_entities.append((start, end, "WATER_BODY"))

        # 方向识别规则
        direction_pattern = r'\b(North|South|East|West|NE|NW|SE|SW|N|S|E|W)\b'
        for match in re.finditer(direction_pattern, text):
            start, end = match.span()
            candidate_entities.append((start, end, "DIRECTION"))

        # 距离识别规则
        distance_pattern = r'(\d+(\.\d+)?)\s*(mi\.|miles|km|meters|m)'
        for match in re.finditer(distance_pattern, text):
            start, end = match.span()
            candidate_entities.append((start, end, "DISTANCE"))

        # 行政区域识别规则
        admin_pattern = r'([A-Z][a-z]+)\s(County|Parish|Township)'
        for match in re.finditer(admin_pattern, text):
            start, end = match.span()
            candidate_entities.append((start, end, "ADMIN_AREA"))

        # 坐标识别规则
        coord_pattern = r'T\d+[NSns]\s?R\d+[EWew]'
        for match in re.finditer(coord_pattern, text):
            start, end = match.span()
            candidate_entities.append((start, end, "COORDINATE"))

        # 相对位置识别规则
        rel_pos_pattern = r'(upstream|downstream|above|below|mouth of|confluence|junction)'
        for match in re.finditer(rel_pos_pattern, text, re.IGNORECASE):
            start, end = match.span()
            candidate_entities.append((start, end, "RELATIVE_POSITION"))

        # 解决实体重叠问题
        # 按长度排序（优先选择更长的实体）
        candidate_entities.sort(key=lambda x: (x[0], -x[1] + x[0]))

        # 使用贪婪算法选择非重叠实体
        selected_entities = []
        covered_indices = set()

        for start, end, label in candidate_entities:
            # 检查是否与已选实体重叠
            overlap = False
            for i in range(start, end):
                if i in covered_indices:
                    overlap = True
                    break

            # 如果没有重叠，添加此实体
            if not overlap:
                selected_entities.append((start, end, label))
                for i in range(start, end):
                    covered_indices.add(i)

        return selected_entities

    def train_model(self, training_data, output_dir, n_iter=100):
        """
        训练spaCy NER模型 (适用于spaCy 3.0+)
        """
        print("开始训练NER模型...")

        # 创建一个新的spaCy模型
        nlp = spacy.blank("en")

        # 添加NER组件
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")

        # 添加实体标签
        for label in self.labels:
            ner.add_label(label)

        # 准备训练数据
        train_examples = []
        for text, annotations in training_data:
            doc = nlp.make_doc(text)

            # 将字符偏移转换为spacy格式的实体
            entities = []
            for start, end, label in annotations["entities"]:
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    entities.append(span)

            # 创建临时Doc以检查实体是否有效
            try:
                temp_doc = nlp.make_doc(text)
                temp_doc.ents = entities

                # 如果没有异常，添加到示例
                example = spacy.training.Example.from_dict(doc, {"entities": annotations["entities"]})
                train_examples.append(example)
            except Exception as e:
                print(f"跳过无效示例: {text}")
                print(f"错误: {e}")

        print(f"有效训练示例数量: {len(train_examples)}")

        # 禁用其他管道
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            # 初始化权重
            if len(train_examples) > 0:
                nlp.begin_training()

                # 训练循环
                for i in range(n_iter):
                    # 随机打乱训练数据
                    random.shuffle(train_examples)
                    losses = {}

                    # 批处理训练
                    batches = spacy.util.minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
                    for batch in batches:
                        nlp.update(batch, drop=0.5, losses=losses)

                    if i % 10 == 0:
                        print(f"迭代 {i}, 损失: {losses}")
            else:
                print("警告: 没有有效的训练示例，跳过训练")

        # 保存模型
        if output_dir is not None and len(train_examples) > 0:
            nlp.to_disk(output_dir)
            print(f"模型已保存到 {output_dir}")

        self.model = nlp
        return nlp

    def test_model(self, test_texts):
        """
        测试NER模型
        """
        if not self.model:
            print("错误: 模型尚未训练")
            return []

        results = []
        for text in test_texts:
            doc = self.model(text)
            entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            results.append({"text": text, "entities": entities})

        return results


# 4. 主程序
def main():
    # 1. 加载数据
    data_path = "your_locality_data.csv"  # 请替换为实际数据路径
    df = load_and_preprocess_data(data_path)

    # # 使用示例数据
    # data = {
    #     'LocalityString': [
    #         "Grand Lake, in bay on S side of Metzlaar Point; T33,34N R7,8E.",
    #         "Yampa River at Yampa 2",
    #         "Woods Creek at Lexington, trib. to Gauley (North) River.",
    #         "South River 4.9 air mi. E Lexington; 0.5 mi. SW Riverside.",
    #         "Cedar Creek trib. to Clinch River above mouth of Burgess Creek jus",
    #         "Guantanamo Bay, N side of South Toro Cay.",
    #         "White River backwater pool 2 to 3 mi. downstream from Oil Troug",
    #         "Guantanamo Bay, Silinas Island.",
    #         "Long Lake, T8S R13W Secs. 9, 10, 15 & 16.",
    #         "Poplar Creek 2.3 mi. NE Oliver Springs 0.5 mi. above mouth of Cov"
    #     ],
    #     'Drainage': [
    #         "Lake Huron", "[NULL]", "James River", "James River",
    #         "Clinch River", "Caribbean Sea", "White River",
    #         "Caribbean Sea", "Lake Michigan", "Tennessee River"
    #     ],
    #     'Country': [
    #         "United States of America", "United States of America",
    #         "United States of America", "United States of America",
    #         "United States of America", "Cuba", "United States of America",
    #         "Cuba", "United States of America", "United States of America"
    #     ],
    #     'State': [
    #         "Michigan", "Colorado", "Virginia", "Virginia",
    #         "Virginia", "Oriente", "Arkansas", "Oriente",
    #         "Michigan", "Tennessee"
    #     ],
    #     'County': [
    #         "Presque Isle", "Moffat", "Rockbridge", "Rockbridge",
    #         "Russell", "[NULL]", "Independence", "[NULL]",
    #         "Cass", "Anderson"
    #     ],
    #     'Continent': [
    #         "North America", "North America", "North America", "North America",
    #         "North America", "North America", "North America", "North America",
    #         "North America", "North America"
    #     ]
    # }

    # df = pd.DataFrame(data)

    # 2. 构建同义词网络
    synonym_builder = GeoSynonymBuilder()
    synonym_network = synonym_builder.extract_potential_synonyms(df)

    # 保存同义词网络
    synonym_builder.save_to_file("geo_synonym_network.json")

    # 导出为ES格式
    es_synonyms = synonym_builder.export_to_elasticsearch_format()
    with open("es_synonyms.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(es_synonyms))

    print(f"导出了 {len(es_synonyms)} 条ES格式同义词")

    # 3. 训练NER模型
    ner_trainer = GeoNERTrainer()
    training_data = ner_trainer.create_training_data(df)

    # 如果数据足够，训练模型
    if len(training_data) > 10:  # 示例中数据很少，实际应用中需要更多数据
        model = ner_trainer.train_model(training_data, "./geo_ner_model")

        # 测试模型
        test_texts = [
            "Mississippi River, left bank at River mile 86.4",
            "Red River, 4 miles south of Bogalusa",
            "Lake Michigan, eastern shore near Saugatuck"
        ]

        test_results = ner_trainer.test_model(test_texts)
        print("\n测试结果:")
        for result in test_results:
            print(f"文本: {result['text']}")
            print(f"识别实体: {result['entities']}\n")
    else:
        print("警告: 训练数据不足，跳过模型训练")

    print("处理完成")


if __name__ == "__main__":
    main()