import elasticsearch
from elasticsearch import Elasticsearch
from typing import Optional, List, Dict, Any


def search_geo_locations(
    query=None,
    field_no=None,
    locality_string=None,
    drainage=None,
    country=None,
    state=None,
    county=None,
    continent=None,
    island=None,
    island_group=None,
    waterbody=None,
    start_date=None,
    inventory=None,
    verbatim_collectors=None,
    fuzzy=True,
    limit=20,
    es_host="localhost",
    es_port=9200,
    index_name="geo_gazetteer"
):
    """
    增强的地理位置搜索函数

    参数:
    - query: 通用搜索查询字符串（搜索所有文本字段）
    - field_no, locality_string, drainage等: 特定字段搜索
    - fuzzy: 启用模糊匹配
    - limit: 最大结果数
    - es_host/es_port/index_name: ES连接参数

    返回:
    - 匹配位置的列表，按相关性排序
    """
    # 连接到Elasticsearch
    es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': 'http'}])

    # 构建搜索查询
    search_query = {
        "size": limit,
        "query": {
            "bool": {
                "should": [],  # 匹配条件
                "filter": [],  # 过滤条件
                "minimum_should_match": 0  # 如果有should条件，至少匹配一个
            }
        },
        "highlight": {
            "fields": {
                "locality_string": {},
                "drainage": {},
                "country": {},
                "state": {},
                "county": {},
                "continent": {},
                "island": {},
                "island_group": {},
                "waterbody": {},
                "full_text": {},
                "verbatim_collectors": {}
            },
            "pre_tags": ["<strong>"],
            "post_tags": ["</strong>"]
        }
    }

    # 处理全文查询
    if query:
        search_query["query"]["bool"]["should"].extend([
            # 全文搜索
            {
                "match": {
                    "full_text": {
                        "query": query,
                        "fuzziness": "AUTO" if fuzzy else "0",
                        "boost": 1.0
                    }
                }
            },
            # 地点描述搜索（给予更高权重）
            {
                "match": {
                    "locality_string": {
                        "query": query,
                        "fuzziness": "AUTO" if fuzzy else "0",
                        "boost": 3.0
                    }
                }
            }
        ])
        search_query["query"]["bool"]["minimum_should_match"] = 1

    # 添加特定字段的搜索条件
    field_queries = {
        "field_no": field_no,
        "locality_string": locality_string,
        "drainage": drainage,
        "country": country,
        "state": state,
        "county": county,
        "continent": continent,
        "island": island,
        "island_group": island_group,
        "waterbody": waterbody,
        "start_date": start_date,
        "inventory": inventory,
        "verbatim_collectors": verbatim_collectors
    }

    for field, value in field_queries.items():
        if value:
            if field in ["field_no"]:  # 精确匹配字段
                search_query["query"]["bool"]["filter"].append(
                    {"term": {field: value}}
                )
            elif field in ["start_date"]:  # 日期字段（现在是文本类型）
                search_query["query"]["bool"]["filter"].append(
                    {"match_phrase": {field: value}}
                )
            else:  # 文本字段，支持模糊匹配
                search_query["query"]["bool"]["should"].append(
                    {
                        "match": {
                            field: {
                                "query": value,
                                "fuzziness": "AUTO" if fuzzy else "0",
                                "boost": 2.0
                            }
                        }
                    }
                )
                search_query["query"]["bool"]["minimum_should_match"] = 1

    # 添加聚合
    search_query["aggs"] = {
        "countries": {
            "terms": {"field": "country.keyword", "size": 10}
        },
        "states": {
            "terms": {"field": "state.keyword", "size": 10}
        },
        "counties": {
            "terms": {"field": "county.keyword", "size": 10}
        },
        "drainages": {
            "terms": {"field": "drainage.keyword", "size": 10}
        },
        "continents": {
            "terms": {"field": "continent.keyword", "size": 10}
        },
        "waterbodies": {
            "terms": {"field": "waterbody.keyword", "size": 10}
        }
    }

    # 如果没有查询条件，返回排名靠前的结果
    if not query and not any(field_queries.values()):
        search_query["query"] = {"match_all": {}}
        search_query["sort"] = [{"_score": {"order": "desc"}}]

    try:
        # 执行搜索
        search_results = es.search(index=index_name, body=search_query)

        # 处理结果
        results = {
            "hits": [],
            "aggregations": {},
            "total": search_results["hits"]["total"]["value"] if "hits" in search_results and "total" in search_results["hits"] else 0
        }

        # 提取匹配结果
        for hit in search_results["hits"]["hits"]:
            source = hit["_source"]
            result = {
                "score": hit["_score"],
                "id": hit["_id"]
            }

            # 复制所有源字段
            for key, value in source.items():
                result[key] = value

            # 添加高亮内容
            if "highlight" in hit:
                result["highlights"] = hit["highlight"]

            results["hits"].append(result)

        # 处理聚合结果
        if "aggregations" in search_results:
            for agg_name, agg_data in search_results["aggregations"].items():
                if "buckets" in agg_data:
                    results["aggregations"][agg_name] = [
                        {"key": bucket["key"], "count": bucket["doc_count"]}
                        for bucket in agg_data["buckets"]
                    ]

        return results
    except Exception as e:
        print(f"搜索时出错: {e}")
        import traceback
        traceback.print_exc()
        return {"hits": [], "aggregations": {}, "total": 0, "error": str(e)}


# 示例使用
if __name__ == "__main__":
    # 示例1：全文搜索
    results = search_geo_locations(query="River Michigan")

    print(f"找到 {results['total']} 个结果:")
    for result in results["hits"][:5]:  # 只显示前5个结果
        print(f"位置: {result.get('locality_string', '-')}")
        print(f"  层级: {result.get('county', '-')}, {result.get('state', '-')}, {result.get('country', '-')}")
        if "drainage" in result and result["drainage"]:
            print(f"  水系: {result['drainage']}")
        print(f"  分数: {result['score']}")
        print()

    # 示例2：特定字段搜索
    results = search_geo_locations(state="Michigan", waterbody="Lake")

    print(f"\n在Michigan州找到的湖泊: {results['total']} 个")
    for result in results["hits"][:5]:
        print(f"位置: {result.get('locality_string', '-')}")
        print(f"  水体: {result.get('waterbody', '-')}")
        print()

    # 示例3：日期搜索
    results = search_geo_locations(
        country="United States",
        start_date="1950"
    )

    print(f"\n1950年在美国的记录: {results['total']} 个")
    for result in results["hits"][:5]:
        print(f"位置: {result.get('locality_string', '-')}")
        print(f"  日期: {result.get('start_date', '-')}")
        print()

    # 显示聚合统计
    if "aggregations" in results and "states" in results["aggregations"]:
        print("\n州/省分布:")
        for state in results["aggregations"]["states"]:
            print(f"{state['key']}: {state['count']} 个结果")