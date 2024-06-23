# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2019.
"""

"""
from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkcore.http.http_config import HttpConfig

from kg.services.kbqa_request import RunKbqaRequest, KbqaRequest
from kg.services.kg_client import KgClient
from kg.services.kg_region import KgRegion


class KgService:
    def __init__(self, ak, sk, config):
        self.ak = ak
        self.sk = sk
        self.config = config

        credentials = BasicCredentials(ak, sk)
        self.client = KgClient.new_builder() \
            .with_http_config(config) \
            .with_credentials(credentials) \
            .with_region(KgRegion.value_of("cn-north-4")) \
            .build()

    def kbqa(self, query, kg_id, session_id=None):
        try:
            request = RunKbqaRequest()
            request.body = KbqaRequest(
                query=query,
                session_id=session_id
            )
            response = self.client.run_kbqa(request, kg_id)
            return response.to_dict()
        except exceptions.ClientRequestException as e:
            print(e.status_code)
            print(e.request_id)
            print(e.error_code)
            print(e.error_msg)
            return None


if __name__ == '__main__':
    print("Begin")
    # 必要配置，配置您的ak、sk信息
    ak = "Your AK"
    sk = "Your SK"
    config = HttpConfig.get_default_config()
    # 网络设置，如您无需设置代理或跳过证书校验，本配置可跳过或删除部分配置项
    # 证书校验，默认为False，需要跳过验证则设为True
    config.ignore_ssl_verification = True
    # 代理账号设置
    config.proxy_protocol = 'http'
    config.proxy_host = 'proxy.xxx.com'
    config.proxy_port = 8080
    config.proxy_user = 'user'
    config.proxy_password = 'password'

    # 初始化机器翻译接口的service
    kg_service = KgService(ak, sk, config)

    print(kg_service.kbqa("国泰中证畜牧养殖ETF的近1月收益率？",
                          kg_id="Your KG ID"))

    print("End")
