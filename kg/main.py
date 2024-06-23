# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2019.
"""

"""

from huaweicloudsdkcore.http.http_config import HttpConfig

from commons.utils import read_objs4json_file, write_objs2json_file
from services.evaluate_service import evaluate
from services.kg_service import KgService


def answer_the_question(kg_service, kg_id, query):
    result = kg_service.kbqa(query,
                             kg_id=kg_id)
    if result["type"] == 2:
        print(result)
    answer = result["frame"][0]["answer"]
    return answer


if __name__ == '__main__':
    print("Begin")
    # 必要配置，配置您的ak、sk信息
    ak = "OK1EHC33LJYBIOD23GFE"
    sk = "Vgjm9WjtE41nSM13lASNCKlFpJuzJ5IVpiauMwEg"
    config = HttpConfig.get_default_config()
    # 网络设置，如您无需设置代理或跳过证书校验，本配置可跳过或删除部分配置项
    # 证书校验，默认为False，需要跳过验证则设为True
    config.ignore_ssl_verification = True
    # 代理账号设置
    #config.proxy_protocol = 'http'
    #config.proxy_host = 'proxy.xxx.com'
    #config.proxy_port = 8080
    #config.proxy_user = 'user'
    #config.proxy_password = 'password'

    kg_id = "Your KG ID"
    kg_id = "43a4b1c8-6085-491c-b2ad-23795cd5afb4"


    # 初始化机器翻译接口的service
    kg_service = KgService(ak, sk, config)

    queries = read_objs4json_file("data/demo.json")
    result = list()
    for query in queries:
        answer = answer_the_question(kg_service, kg_id, query["query"])
        result.append({
            "id": query["id"],
            "query": query["query"],
            "answer": answer
        })
    write_objs2json_file(result, "data/demo_pred_result.json")

    true_result = read_objs4json_file("data/demo_true_result.json")
    pred_result = read_objs4json_file("data/demo_pred_result.json")
    score = evaluate(true_result, pred_result)
    print("Score: ", score)
    result=str(score)
    with open("result.txt","w") as f:
        f.write(result)
        

    print("End")
