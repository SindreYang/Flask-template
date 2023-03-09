# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：FlaskDeploy -> __init__.py.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2022/12/5 14:31
@Version: V0.1
@License: (C)Copyright 2021-2022 , UP3D
@Reference: 
@History:
- 2022/12/5 :
==================================================
'''
__author__ = 'sindre'
from  conf.config import split_config,generate_config,landmark_config,split_and_generate_config
from flask import Flask, request, make_response, send_from_directory, send_file, Response,Blueprint, jsonify
from models.Split_deploy import *
from models.generate_deploy import generate_mesh
from models.landmark_deploy import landmarks_inference
import os
from . import split,generate,landmark,split_and_generate

# 注册蓝图
def init_app(app):
    app.register_blueprint(split.blueprint)
    app.register_blueprint(generate.blueprint)
    app.register_blueprint(landmark.blueprint)
    app.register_blueprint(split_and_generate.blueprint)

