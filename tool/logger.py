"""
全流程训练日志，支持四阶段标记：[DATA]/[FEAT]/[TRAIN]/[EVAL]。
"""

import os
import datetime


# ANSI 颜色码
_COLORS = {

}


class Logger:
    """
    全流程日志记录器, 数据处理/质量检查-特征构建-模型训练-测试评估, log文件的时间戳后缀命名的粒度为天, 一天只产生一个log文件。
    """

    def __init__(self, log_dir, model_name):
       pass