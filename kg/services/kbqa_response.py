# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2019.
"""

"""
import re
import six


from huaweicloudsdkcore.sdk_response import SdkResponse
from huaweicloudsdkcore.utils.http_utils import sanitize_for_serialization


class KbqaResponse(SdkResponse):

    """
    Attributes:
      openapi_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """

    sensitive_list = []

    openapi_types = {
        'session_id': 'str',
        'type': 'int',
        'frame': 'list[object]',
        'error_code': 'str',
        'error_msg': 'str'
    }

    attribute_map = {
        'session_id': 'session_id',
        'type': 'type',
        'frame': 'frame',
        'error_code': 'error_code',
        'error_msg': 'error_msg'
    }

    def __init__(self, answer=None, session_id=None, type=None, frame=None, error_code=None, error_msg=None):
        """RunAspectSentimentResponse

        The model defined in huaweicloud sdk

        :param text: 待分析文本
        :type text: str
        :param label: 该文本的整体情感标签，取值如下： 0  负向 1  正向
        :type label: int
        :param confidence: 该文本整体情感label的置信度,小数点精确到3位。
        :type confidence: float
        :param aspect_opinions: 属性级情感挖掘列表
        :type aspect_opinions: list[:class:`huaweicloudsdknlp.v2.AspectOpinion`]
        :param error_code: 调用失败时的错误码，具体请参见错误码。调用成功时无此字段。
        :type error_code: str
        :param error_msg: 调用失败时的错误信息。调用成功时无此字段。
        :type error_msg: str
        """

        super(KbqaResponse, self).__init__()

        self._answer = None
        self._session_id = None
        self._type = None
        self._frame = None
        self._error_code = None
        self._error_msg = None
        self.discriminator = None

        if answer is not None:
            self.text = answer
        if session_id is not None:
            self.session_id = session_id
        if type is not None:
            self.type = type
        if frame is not None:
            self.frame = frame
        if error_code is not None:
            self.error_code = error_code
        if error_msg is not None:
            self.error_msg = error_msg

    @property
    def answer(self):
        """Gets the text of this RunAspectSentimentResponse.

        待分析文本

        :return: The text of this RunAspectSentimentResponse.
        :rtype: str
        """
        return self._answer

    @answer.setter
    def answer(self, answer):
        """Sets the text of this RunAspectSentimentResponse.

        待分析文本

        :param text: The text of this RunAspectSentimentResponse.
        :type text: str
        """
        self._text = answer

    @property
    def session_id(self):
        """Gets the label of this RunAspectSentimentResponse.

        该文本的整体情感标签，取值如下： 0  负向 1  正向

        :return: The label of this RunAspectSentimentResponse.
        :rtype: int
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id):
        """Sets the label of this RunAspectSentimentResponse.

        该文本的整体情感标签，取值如下： 0  负向 1  正向

        :param label: The label of this RunAspectSentimentResponse.
        :type label: int
        """
        self._session_id = session_id

    @property
    def type(self):
        """Gets the confidence of this RunAspectSentimentResponse.

        该文本整体情感label的置信度,小数点精确到3位。

        :return: The confidence of this RunAspectSentimentResponse.
        :rtype: float
        """
        return self._type

    @type.setter
    def type(self, type):
        """Sets the confidence of this RunAspectSentimentResponse.

        该文本整体情感label的置信度,小数点精确到3位。

        :param confidence: The confidence of this RunAspectSentimentResponse.
        :type confidence: float
        """
        self._type = type

    @property
    def frame(self):
        """Gets the aspect_opinions of this RunAspectSentimentResponse.

        属性级情感挖掘列表

        :return: The aspect_opinions of this RunAspectSentimentResponse.
        :rtype: list[:class:`huaweicloudsdknlp.v2.AspectOpinion`]
        """
        return self._frame

    @frame.setter
    def frame(self, frame):
        """Sets the aspect_opinions of this RunAspectSentimentResponse.

        属性级情感挖掘列表

        :param aspect_opinions: The aspect_opinions of this RunAspectSentimentResponse.
        :type aspect_opinions: list[:class:`huaweicloudsdknlp.v2.AspectOpinion`]
        """
        self._frame = frame

    @property
    def error_code(self):
        """Gets the error_code of this RunAspectSentimentResponse.

        调用失败时的错误码，具体请参见错误码。调用成功时无此字段。

        :return: The error_code of this RunAspectSentimentResponse.
        :rtype: str
        """
        return self._error_code

    @error_code.setter
    def error_code(self, error_code):
        """Sets the error_code of this RunAspectSentimentResponse.

        调用失败时的错误码，具体请参见错误码。调用成功时无此字段。

        :param error_code: The error_code of this RunAspectSentimentResponse.
        :type error_code: str
        """
        self._error_code = error_code

    @property
    def error_msg(self):
        """Gets the error_msg of this RunAspectSentimentResponse.

        调用失败时的错误信息。调用成功时无此字段。

        :return: The error_msg of this RunAspectSentimentResponse.
        :rtype: str
        """
        return self._error_msg

    @error_msg.setter
    def error_msg(self, error_msg):
        """Sets the error_msg of this RunAspectSentimentResponse.

        调用失败时的错误信息。调用成功时无此字段。

        :param error_msg: The error_msg of this RunAspectSentimentResponse.
        :type error_msg: str
        """
        self._error_msg = error_msg

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.openapi_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                if attr in self.sensitive_list:
                    result[attr] = "****"
                else:
                    result[attr] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        import simplejson as json
        if six.PY2:
            import sys
            reload(sys)
            sys.setdefaultencoding("utf-8")
        return json.dumps(sanitize_for_serialization(self), ensure_ascii=False)

    def __repr__(self):
        """For `print`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, KbqaResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
