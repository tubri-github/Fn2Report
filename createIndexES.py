import os
import json
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
            self.es = Elasticsearch([{'host': self.es_host, 'port': self.es_port, 'scheme': 'http'}])
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
        """创建具有正确同义词配置的Elasticsearch索引"""
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
                    "name": {
                        "type": "text",
                        "analyzer": "geo_analyzer",
                        "search_analyzer": "geo_search_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "completion": {
                                "type": "completion"  # 不使用contexts
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
                    "source_text": {"type": "text"}
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

    def load_original_texts(self, file_path="geo_locations.txt"):
        """加载原始文本数据"""
        self.original_texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if ' with entities "' in line:
                    text = line.strip().split(' with entities "')[0].strip('"')
                    self.original_texts.append(text)
                else:
                    self.original_texts.append(line.strip())

        print(f"加载了 {len(self.original_texts)} 条原始文本")
        return self

    def _prepare_synonym_file(self, output_path="analysis/synonyms.txt"):
        """为Elasticsearch准备同义词文件 (保留但不再使用)"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for category, synonym_groups in self.gazetteer_data["synonyms"].items():
                for key_word, synonyms in synonym_groups.items():
                    if synonyms:
                        # 格式: term1, term2, term3 => term1, term2, term3
                        synonym_line = f"{key_word}, {', '.join(synonyms)} => {key_word}, {', '.join(synonyms)}"
                        f.write(synonym_line + "\n")

        print(f"同义词文件已准备好，位置：{output_path}")
        return output_path

    def _get_inline_synonyms(self):
        """从同义词数据生成内联同义词列表"""
        synonyms_list = []

        for category, synonym_groups in self.gazetteer_data["synonyms"].items():
            for key_word, synonyms in synonym_groups.items():
                if synonyms:
                    # 格式: "term1, term2, term3 => term1, term2, term3"
                    synonym_line = f"{key_word}, {', '.join(synonyms)} => {key_word}, {', '.join(synonyms)}"
                    synonyms_list.append(synonym_line)

        return synonyms_list

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
                            "source_text": example_texts[:3] if example_texts else []
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
                        "source_text": [text]
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
                    # 修改这里：使用stats_only=True，只返回成功和失败的数量
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

                    # 修改这里：错误数是整数，而不是列表
                    total_failed += len(e.errors)
                except Exception as e:
                    print(f"  批次处理错误: {e}")
                    # 修改这里：整个批次失败时，失败数增加批次大小
                    total_failed += len(batch)

            print(f"索引完成: 成功 {total_success} 个文档, 失败 {total_failed} 个")
            return True
        except Exception as e:
            print(f"批量索引期间出错: {e}")
            return False

    def create_search_function(self, output_file="geo_search.py"):
        """创建增强的搜索函数"""
        search_function = """
import elasticsearch
from elasticsearch import Elasticsearch
from typing import Optional, List, Dict, Any

def search_geo_locations(query, categories=None, subcategories=None, fuzzy=True, limit=20, 
                        es_host="localhost", es_port=9200, index_name="geo_gazetteer"):
    \"\"\"
    增强的地理位置搜索函数

    参数:
    - query: 搜索查询字符串
    - categories: 可选的类别过滤列表 
    - subcategories: 可选的子类别过滤列表
    - fuzzy: 启用模糊匹配
    - limit: 最大结果数
    - es_host/es_port/index_name: ES连接参数

    返回:
    - 匹配位置的列表，按相关性分组
    \"\"\"
    # 连接到Elasticsearch
    es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])

    # 分析查询，识别可能的地理实体
    query_parts = query.split()

    # 构建多字段搜索查询
    search_query = {
        "size": limit,
        "query": {
            "bool": {
                "should": [
                    # 全文搜索 - 主要匹配方式
                    {
                        "match": {
                            "full_text": {
                                "query": query,
                                "fuzziness": "AUTO" if fuzzy else "0",
                                "boost": 2.0  # 给予全文匹配更高权重
                            }
                        }
                    },
                    # 名称精确匹配
                    {
                        "match": {
                            "name": {
                                "query": query,
                                "fuzziness": "AUTO" if fuzzy else "0",
                                "boost": 3.0  # 给予名称匹配最高权重
                            }
                        }
                    },
                    # 源文本匹配
                    {
                        "match": {
                            "source_text": {
                                "query": query,
                                "fuzziness": "AUTO" if fuzzy else "0"
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "highlight": {
            "fields": {
                "name": {},
                "full_text": {},
                "source_text": {}
            },
            "pre_tags": ["<strong>"],
            "post_tags": ["</strong>"]
        }
    }

    # 添加类别和子类别过滤器（如果提供）
    if categories or subcategories:
        search_query["query"]["bool"]["filter"] = []

        if categories:
            if isinstance(categories, str):
                categories = [categories]
            search_query["query"]["bool"]["filter"].append({
                "terms": {"category": categories}
            })

        if subcategories:
            if isinstance(subcategories, str):
                subcategories = [subcategories]
            search_query["query"]["bool"]["filter"].append({
                "terms": {"subcategory": subcategories}
            })

    # 添加聚合以按类别和子类别分组
    search_query["aggs"] = {
        "category_group": {
            "terms": {
                "field": "category",
                "size": 10
            },
            "aggs": {
                "subcategory_group": {
                    "terms": {
                        "field": "subcategory",
                        "size": 10
                    }
                }
            }
        }
    }

    # 执行搜索
    search_results = es.search(index=index_name, body=search_query)

    # 处理结果
    results = {
        "hits": [],
        "categories": {},
        "suggested_queries": []
    }

    # 提取匹配结果
    for hit in search_results["hits"]["hits"]:
        result = {
            "name": hit["_source"]["name"],
            "category": hit["_source"]["category"],
            "subcategory": hit["_source"]["subcategory"],
            "score": hit["_score"]
        }

        # 添加高亮内容
        if "highlight" in hit:
            result["highlights"] = hit["highlight"]

        # 添加示例文本
        if "source_text" in hit["_source"] and hit["_source"]["source_text"]:
            result["examples"] = hit["_source"]["source_text"]

        results["hits"].append(result)

    # 提取类别和子类别聚合
    if "aggregations" in search_results:
        for cat_bucket in search_results["aggregations"]["category_group"]["buckets"]:
            cat_name = cat_bucket["key"]
            results["categories"][cat_name] = {
                "count": cat_bucket["doc_count"],
                "subcategories": {}
            }

            for subcat_bucket in cat_bucket["subcategory_group"]["buckets"]:
                subcat_name = subcat_bucket["key"]
                subcat_count = subcat_bucket["doc_count"]
                results["categories"][cat_name]["subcategories"][subcat_name] = subcat_count

    # 生成建议查询
    if query_parts and len(query_parts) > 1:
        # 针对多词查询，建议使用不同组合
        for i in range(len(query_parts)):
            suggested_query = " ".join(query_parts[:i] + query_parts[i+1:])
            results["suggested_queries"].append(suggested_query)

    return results

# 示例使用
if __name__ == "__main__":
    # 示例1：搜索"Missisippi River"（故意拼错）
    results = search_geo_locations("Missisippi River", fuzzy=True)

    print(f"找到 {len(results['hits'])} 个结果:")
    for result in results["hits"][:5]:  # 只显示前5个结果
        print(f"{result['name']} ({result['category']}/{result['subcategory']}) - 分数: {result['score']}")
        if "examples" in result and result["examples"]:
            print(f"  示例: {result['examples'][0]}")
        print()

    # 示例2：混合查询"Missisippi creek"
    results = search_geo_locations("Missisippi creek")

    print(f"\\n混合查询找到 {len(results['hits'])} 个结果:")
    for result in results["hits"][:5]:
        print(f"{result['name']} ({result['category']}/{result['subcategory']}) - 分数: {result['score']}")
        print()

    # 示例3：按类别过滤
    results = search_geo_locations("river", categories=["water_bodies"])

    print(f"\\n在water_bodies类别中找到 {len(results['hits'])} 个river:")
    for result in results["hits"][:5]:
        print(f"{result['name']} ({result['subcategory']}) - 分数: {result['score']}")
        print()

    # 示例4：完整地址搜索
    complex_query = "3 miles north of Mississippi River near Highway 61"
    results = search_geo_locations(complex_query)

    print(f"\\n复杂查询找到 {len(results['hits'])} 个结果:")
    for result in results["hits"][:5]:
        print(f"{result['name']} ({result['category']}/{result['subcategory']}) - 分数: {result['score']}")
        if "highlights" in result:
            print(f"  匹配: {result['highlights']}")
        print()

    # 类别统计
    if results["categories"]:
        print("\\n类别分布:")
        for cat, info in results["categories"].items():
            print(f"{cat}: {info['count']} 个结果")
            for subcat, count in info["subcategories"].items():
                print(f"  - {subcat}: {count}")

    # 建议查询
    if results["suggested_queries"]:
        print("\\n您可能还想搜索:")
        for suggestion in results["suggested_queries"]:
            print(f"- {suggestion}")
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(search_function)

        print(f"创建了增强的搜索函数，保存在 {output_file}")
        return self


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

    # 创建搜索函数
    indexer.create_search_function()

    print("索引过程完成！")


if __name__ == "__main__":
    index_geo_gazetteer()