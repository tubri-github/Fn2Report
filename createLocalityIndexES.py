import os
import json
import ssl

import elasticsearch
from elasticsearch import Elasticsearch, helpers
import pandas as pd


class GeoElasticSearchIndexer:
    """地理位置词典的Elasticsearch索引器"""

    def __init__(self, es_host="localhost", es_port=9200, index_name="geo_gazetteer"):
        """
        初始化Elasticsearch索引器

        参数:
        - es_host: Elasticsearch主机地址
        - es_port: Elasticsearch端口
        - index_name: 要创建的索引名称
        """
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = index_name
        self.es = None
        self.gazetteer_data = {}

    def connect_elasticsearch(self):
        """连接到Elasticsearch"""
        try:
            # 包含scheme参数的连接方法
            # 创建 SSL 上下文（不验证证书）
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            self.es = Elasticsearch(
                hosts=[{
                    'host': self.es_host,
                    'port': self.es_port,
                    'scheme': 'https'  # 使用 https
                }],
                basic_auth=(self.es_user, self.es_password),
                ssl_context=context,
                verify_certs=False,
                ssl_show_warn=False
            )

            if self.es.ping():
                print("成功连接到Elasticsearch")
                return True
            else:
                print("无法连接到Elasticsearch")
                return False
        except elasticsearch.exceptions.ConnectionError:
            print("Elasticsearch连接错误")
            return False

    def create_index(self, delete_if_exists=True):
        """创建具有正确同义词配置的Elasticsearch索引，支持多字段搜索"""
        if delete_if_exists:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                print(f"删除现有索引: {self.index_name}")

        # 首先从同义词JSON生成Elasticsearch同义词列表
        synonyms_list = []

        # 加载同义词数据
        synonym_path = os.path.join("gazetteers", "synonyms.json")
        if os.path.exists(synonym_path):
            with open(synonym_path, "r", encoding="utf-8") as f:
                synonyms_data = json.load(f)

                # 处理每个类别的同义词
                for category, synonym_groups in synonyms_data.items():
                    for key_word, synonyms in synonym_groups.items():
                        if synonyms:
                            # 同义词格式: term1, term2, term3 => term1, term2, term3
                            all_terms = [key_word] + synonyms
                            synonym_line = f"{', '.join(all_terms)} => {', '.join(all_terms)}"
                            synonyms_list.append(synonym_line)

        print(f"从JSON生成了 {len(synonyms_list)} 条同义词规则")

        # 创建索引设置
        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "geo_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "geo_synonym_filter", "edge_ngram_filter"]
                        },
                        "geo_search_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "geo_synonym_filter"]
                        }
                    },
                    "filter": {
                        "edge_ngram_filter": {
                            "type": "edge_ngram",
                            "min_gram": 2,
                            "max_gram": 20
                        },
                        "geo_synonym_filter": {
                            "type": "synonym",
                            "synonyms": synonyms_list  # 使用生成的同义词列表
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # 原有字段保持不变
                    "name": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "completion": {
                                "type": "completion"  # 自动完成
                            }
                        }
                    },
                    "full_text": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer"
                    },
                    "category": {"type": "keyword"},
                    "subcategory": {"type": "keyword"},
                    "count": {"type": "integer"},
                    "synonyms": {"type": "text", "analyzer": "geo_analyzer"},
                    "related_terms": {"type": "text", "analyzer": "geo_analyzer"},
                    "source_text": {"type": "text"},

                    # CSV数据中的字段
                    "locality_id": {"type": "keyword"},  # serial4类型，用keyword存储
                    "field_no": {  "type": "text",
                              "fields": {
                                "keyword": {
                                  "type": "keyword"
                                }
                              }},  # varchar(50)类型
                    "locality_string": {  # 保持原有设置
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "raw": {
                                "type": "text",
                                "analyzer": "standard"
                            }
                        }
                    },
                    "drainage": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "country": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "state": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "county": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "continent": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "island": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "island_group": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "elevation_method": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "waterbody": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "start_date": {
                        "type": "text",  # 改为text类型，避免日期格式问题
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "inventory": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "verbatim_collectors": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    }
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=settings)
            print(f"创建索引: {self.index_name}")
            return True
        except Exception as e:
            print(f"创建索引出错: {e}")
            return False

    def load_gazetteer_data(self, gazetteer_dir="gazetteers"):
        """从文件加载词典数据"""
        # 加载主要词典
        self.gazetteer_data["primary"] = {}
        for filename in os.listdir(gazetteer_dir):
            if filename.endswith(".txt") and not filename.startswith("synonyms"):
                category = filename.replace(".txt", "")
                self.gazetteer_data["primary"][category] = []

                with open(os.path.join(gazetteer_dir, filename), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            name, count = parts[0], int(parts[1])
                            self.gazetteer_data["primary"][category].append({"name": name, "count": count})

        # 加载层次词典
        self.gazetteer_data["hierarchical"] = {}
        for category in os.listdir(gazetteer_dir):
            category_dir = os.path.join(gazetteer_dir, category)
            if os.path.isdir(category_dir):
                self.gazetteer_data["hierarchical"][category] = {}

                for filename in os.listdir(category_dir):
                    if filename.endswith(".txt"):
                        subcategory = filename.replace(".txt", "")
                        self.gazetteer_data["hierarchical"][category][subcategory] = []

                        with open(os.path.join(category_dir, filename), "r", encoding="utf-8") as f:
                            for line in f:
                                name = line.strip()
                                self.gazetteer_data["hierarchical"][category][subcategory].append(name)

        # 加载同义词数据
        self.gazetteer_data["synonyms"] = {}
        synonym_path = os.path.join(gazetteer_dir, "synonyms.json")
        if os.path.exists(synonym_path):
            with open(synonym_path, "r", encoding="utf-8") as f:
                self.gazetteer_data["synonyms"] = json.load(f)

        # 加载模式
        pattern_path = os.path.join(gazetteer_dir, "patterns.json")
        if os.path.exists(pattern_path):
            with open(pattern_path, "r", encoding="utf-8") as f:
                self.gazetteer_data["patterns"] = json.load(f)

        print(f"从 {gazetteer_dir} 加载了词典数据")
        return self


    def index_gazetteer_data(self):
        """将改进的词典数据索引到Elasticsearch，添加错误处理"""
        if not self.es or not self.gazetteer_data:
            print("Elasticsearch连接或词典数据不可用")
            return False

        # 准备批量索引数据
        bulk_data = []

        # 处理主要词典条目
        for category, entries in self.gazetteer_data["primary"].items():
            for entry in entries:
                try:
                    # 查找相关术语
                    related_terms = []
                    if category in self.gazetteer_data["synonyms"]:
                        for key_word, synonyms in self.gazetteer_data["synonyms"][category].items():
                            if key_word.lower() in entry["name"].lower():
                                related_terms.extend(synonyms)

                    # 查找示例文本
                    example_texts = []
                    for text in self.original_texts:
                        if entry["name"] in text:
                            example_texts.append(text)
                            if len(example_texts) >= 3:  # 限制为3个示例
                                break

                    # 确定子类别
                    subcategory = "general"
                    if category in self.gazetteer_data["hierarchical"]:
                        for subcat, subcat_entries in self.gazetteer_data["hierarchical"][category].items():
                            if entry["name"] in subcat_entries:
                                subcategory = subcat
                                break

                    # 创建组合全文字段
                    full_text_parts = [
                        entry["name"],  # 名称
                        f"{category} {subcategory}",  # 类别和子类别
                        " ".join(related_terms) if related_terms else "",  # 相关术语
                        " ".join(example_texts) if example_texts else ""  # 示例文本
                    ]
                    full_text = " ".join([part for part in full_text_parts if part])

                    # 生成安全的ID
                    safe_name = entry["name"].replace(' ', '_').replace('/', '_').replace('\\', '_')
                    doc_id = f"{category}_{safe_name}"

                    # 准备文档
                    doc = {
                        "_index": self.index_name,
                        "_id": doc_id,
                        "_source": {
                            "name": entry["name"],
                            "full_text": full_text,  # 新增组合字段
                            "category": category,
                            "subcategory": subcategory,
                            "count": entry["count"],
                            "synonyms": related_terms,
                            "related_terms": related_terms,
                            "source_text": example_texts[:3] if example_texts else [],
                            "locality_string": entry["name"]  # 将name作为locality_string
                        }
                    }

                    bulk_data.append(doc)
                except Exception as e:
                    print(f"处理词典条目时出错: {category} - {entry['name']}: {e}")

        # 索引原始文本数据
        for i, text in enumerate(self.original_texts):
            try:
                # 提取可能的类别和地点
                categories = []
                for category, entries in self.gazetteer_data["primary"].items():
                    for entry in entries:
                        if entry["name"] in text:
                            categories.append(f"{category}:{entry['name']}")

                doc = {
                    "_index": self.index_name,
                    "_id": f"text_{i}",
                    "_source": {
                        "name": text[:50] + "..." if len(text) > 50 else text,
                        "full_text": text,  # 原始文本作为全文
                        "category": "original_text",
                        "subcategory": "text",
                        "count": 1,
                        "related_terms": categories[:10] if categories else [],
                        "source_text": [text],
                        "locality_string": text  # 将文本作为locality_string
                    }
                }
                bulk_data.append(doc)
            except Exception as e:
                print(f"处理原始文本时出错 (index {i}): {e}")

        try:
            # 分批索引，每批1000个文档
            batch_size = 1000
            total_success = 0
            total_failed = 0

            for i in range(0, len(bulk_data), batch_size):
                batch = bulk_data[i:i + batch_size]
                print(f"索引批次 {i // batch_size + 1}/{(len(bulk_data) - 1) // batch_size + 1} (大小: {len(batch)})")

                try:
                    # 使用stats_only=True，只返回成功和失败的数量
                    success, failed = helpers.bulk(self.es, batch, stats_only=True)
                    total_success += success
                    total_failed += failed
                    print(f"  批次结果: 成功 {success} 个文档, 失败 {failed} 个")
                except helpers.BulkIndexError as e:
                    # 查看详细错误信息
                    print(f"  批量索引错误: {len(e.errors)} 个文档失败")

                    # 分析前10个错误
                    for j, error in enumerate(e.errors[:10]):
                        error_item = list(error.items())[0]  # 获取第一个键值对
                        op_type, error_details = error_item

                        if 'error' in error_details:
                            error_type = error_details['error'].get('type', 'unknown')
                            error_reason = error_details['error'].get('reason', 'unknown')
                            print(f"    错误 {j + 1}: 类型={error_type}, 原因={error_reason}")
                        else:
                            print(f"    错误 {j + 1}: {error_details}")

                    # 错误数是整数，而不是列表
                    total_failed += len(e.errors)
                except Exception as e:
                    print(f"  批次处理错误: {e}")
                    # 整个批次失败时，失败数增加批次大小
                    total_failed += len(batch)

            print(f"索引完成: 成功 {total_success} 个文档, 失败 {total_failed} 个")
            return True
        except Exception as e:
            print(f"批量索引期间出错: {e}")
            return False

    def index_data_from_csv(self, csv_file_path):
        """从CSV文件中索引地理位置数据，直接使用CSV中的LocalityString"""
        if not self.es:
            print("Elasticsearch连接不可用")
            return False

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path, delimiter='\t')
            print(f"从CSV加载了 {len(df)} 条记录")

            # 准备批量索引数据
            bulk_data = []

            # 处理CSV中的每一行
            for index, row in df.iterrows():
                try:
                    # 确保字段存在，如果不存在则设为空
                    locality_id = str(row.get('Locality1ID', '')) if pd.notna(row.get('Locality1ID', '')) else ''
                    field_no = str(row.get('FieldNo', '')) if pd.notna(row.get('FieldNo', '')) else ''
                    locality_string = str(row.get('LocalityString', '')) if pd.notna(
                        row.get('LocalityString', '')) else ''
                    drainage = str(row.get('Drainage', '')) if pd.notna(row.get('Drainage', '')) else ''
                    country = str(row.get('Country', '')) if pd.notna(row.get('Country', '')) else ''
                    state = str(row.get('State', '')) if pd.notna(row.get('State', '')) else ''
                    county = str(row.get('County', '')) if pd.notna(row.get('County', '')) else ''
                    continent = str(row.get('Continent', '')) if pd.notna(row.get('Continent', '')) else ''
                    island = str(row.get('Island', '')) if pd.notna(row.get('Island', '')) else ''
                    island_group = str(row.get('Island Group', '')) if pd.notna(row.get('Island Group', '')) else ''
                    elevation_method = str(row.get('ElevationMethod', '')) if pd.notna(
                        row.get('ElevationMethod', '')) else ''
                    waterbody = str(row.get('Waterbody', '')) if pd.notna(row.get('Waterbody', '')) else ''

                    # 处理日期字段 - 转换为字符串并处理空值
                    start_date = ''
                    if pd.notna(row.get('StartDate', None)):
                        start_date_value = row.get('StartDate', '')
                        # 如果是日期类型，转换为yyyy-MM-dd格式的字符串
                        if hasattr(start_date_value, 'strftime'):
                            start_date = start_date_value.strftime('%Y-%m-%d')
                        else:
                            # 否则直接转换为字符串
                            start_date = str(start_date_value)

                    inventory = str(row.get('Inventory', '')) if pd.notna(row.get('Inventory', '')) else ''
                    verbatim_collectors = str(row.get('VerbatimCollectors', '')) if pd.notna(
                        row.get('VerbatimCollectors', '')) else ''

                    # 创建组合全文字段 - 包含所有可能的搜索字段
                    full_text_parts = [
                        locality_string,
                        drainage,
                        country,
                        state,
                        county,
                        continent,
                        island,
                        island_group,
                        waterbody,
                        start_date,
                        inventory,
                        verbatim_collectors
                    ]
                    full_text = " ".join([part for part in full_text_parts if part and part.strip()])

                    # 准备文档
                    doc = {
                        "_index": self.index_name,
                        "_id": locality_id if locality_id else f"loc_{index}",
                        "_source": {
                            "name": locality_string[:50] + "..." if len(locality_string) > 50 else locality_string,
                            # 使用地点描述的开头作为名称
                            "full_text": full_text,
                            "category": "geography",  # 可根据需要设置其他类别
                            "subcategory": "location",  # 可根据需要设置子类别
                            "count": 1,

                            # CSV字段
                            "locality_id": locality_id,
                            "field_no": field_no,
                            "locality_string": locality_string,
                            "drainage": drainage,
                            "country": country,
                            "state": state,
                            "county": county,
                            "continent": continent,
                            "island": island,
                            "island_group": island_group,
                            "elevation_method": elevation_method,
                            "waterbody": waterbody,
                            "start_date": start_date,
                            "inventory": inventory,
                            "verbatim_collectors": verbatim_collectors
                        }
                    }

                    bulk_data.append(doc)

                    # 每1000条记录批量处理一次，减小批次大小以降低错误风险
                    if len(bulk_data) >= 1000:
                        try:
                            helpers.bulk(self.es, bulk_data)
                            print(f"已索引 {index + 1} 条记录")
                        except helpers.BulkIndexError as e:
                            print(f"批量索引错误，失败 {len(e.errors)} 条记录")
                            # 可选：记录第一个错误详情
                            if e.errors:
                                print(f"第一个错误: {e.errors[0]}")
                        bulk_data = []

                except Exception as e:
                    print(f"处理CSV行 {index} 时出错: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误信息

            # 处理剩余的记录
            if bulk_data:
                try:
                    helpers.bulk(self.es, bulk_data)
                    print(f"已索引剩余 {len(bulk_data)} 条记录")
                except helpers.BulkIndexError as e:
                    print(f"批量索引错误，失败 {len(e.errors)} 条记录")
                    if e.errors:
                        print(f"第一个错误: {e.errors[0]}")

            print(f"已完成对 {len(df)} 条记录的处理")
            return True

        except Exception as e:
            print(f"索引CSV数据时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return False

# 主函数来运行索引过程
def index_geo_gazetteer():
    indexer = GeoElasticSearchIndexer()

    # 连接到Elasticsearch
    if not indexer.connect_elasticsearch():
        print("连接到Elasticsearch失败。请确保它正在运行。")
        return

    # 创建索引
    indexer.create_index()

    # 加载词典数据
    indexer.load_gazetteer_data()

    # 加载原始文本数据
    indexer.load_original_texts()

    # 索引数据
    indexer.index_gazetteer_data()

    print("索引过程完成！")


# 主函数来运行索引过程
def index_geo_gazetteer_from_csv(csv_file_path):
    indexer = GeoElasticSearchIndexer()

    # 连接到Elasticsearch
    if not indexer.connect_elasticsearch():
        print("连接到Elasticsearch失败。请确保它正在运行。")
        return

    # 创建索引
    indexer.create_index()

    # 从CSV索引数据
    indexer.index_data_from_csv(csv_file_path)

    print("索引过程完成！")


if __name__ == "__main__":
    # 执行CSV索引
    index_geo_gazetteer_from_csv("locations.csv")  # 请替换为您的CSV文件路径