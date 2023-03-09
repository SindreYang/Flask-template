# -*- coding: UTF-8 -*-
'''
=================================================
@path   ：FlaskDeploy -> split.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2022/12/5 15:27
@Version: V0.1
@License: (C)Copyright 2021-2022 , UP3D
@Reference: 
@History:
- 2022/12/5 :
==================================================
'''
__author__ = 'sindre'

from . import *

blueprint = Blueprint(split_config.name, __name__, url_prefix=f'/{split_config.name}')


@blueprint.route("", methods=['GET', 'POST'])
def Seg():
    print(request.files)
    print(request.form)
    mesh = request.files["file"]
    args = split_config.merger_args
    constraints = int(request.form["constraints"])
    refine_switch = bool(int(request.form["refine_switch"]))
    model_arg = (request.form["model"]).replace(" ","")
    mesh_path = os.path.join(split_config.cache_dir, mesh.filename)
    mesh.save(mesh_path)
    if not mesh:
        return "The mesh  is empty"
    if model_arg in args.keys():
        model_path, class_num = args[model_arg][0], args[model_arg][1]
        T = TeethSeg(mesh_path=mesh_path, model_path=model_path, class_num=class_num, constraints=constraints,
                     refine_switch=refine_switch)
        T.test_simple(save_path=split_config.ply_path, sampled=False)
        export_mtl(import_mesh=split_config.ply_path, save_mesh=split_config.obj_path)
        return "upper is ok", 200
    else:
        return f"model arg not in {args.keys()}", 400


@blueprint.route('/download/<arg>', methods=['GET', 'POST'])
def download(arg):
    if arg == "obj":
        if os.path.exists(split_config.obj_path):
            r = make_response(send_file(split_config.obj_path))
            return r
    if arg == "mtl":
        if os.path.exists(split_config.mtl_path):
            r = make_response(send_file(split_config.mtl_path))
            return r
    if arg == "ply":
        if os.path.exists(split_config.ply_path):
            r = make_response(send_file(split_config.ply_path))
            return r
    return "Not Find FILE", 400
