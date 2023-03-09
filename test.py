# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：FlaskDeploy -> test.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2023/1/3 10:54
@Version: V0.1
@License: (C)Copyright 2021-2022 , UP3D
@Reference: 
@History:
- 2023/1/3 :
==================================================
'''
__author__ = 'sindre'

from models.Split_deploy import TeethSeg

if __name__ == '__main__':
    u = R"C:\Users\sindre\Downloads\上颌 (7).ply"
    m = R'resources/upper_17.pt'
    T = TeethSeg(u, m)
    T.extraction_57()