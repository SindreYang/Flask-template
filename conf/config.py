# 配置
class app_config:
    """
    flask 启动配置
    """
    DEBUG = True
    FLASK_ENV = "development"


class split_config:
    """
    分牙配置
    参数:
    file:文件
    constraints：优化强度
    refine_switch：优化开关
    model：选择运行模型（字符串，如"上颌"）

    上下请求：
    192.168.1.X/split


    数据下载请求：
    192.168.1.X/split/download/mtl;
    192.168.1.X/split/download/obj;
    192.168.1.X/split/download/ply;
    """
    name = "split"  # 主域名
    cache_dir = "resources/cache"  # 上传模型保存路径
    obj_path = "resources/cache/mesh.obj"  # 输出obj文件路径，与mtl匹配
    mtl_path = "resources/cache/mesh.obj.mtl"  # 输出mtl文件路径
    ply_path = "resources/cache/mesh.ply"  # 输出ply文件，带颜色

    merger_args = {"对称分牙上颌": ["resources/upper_9.pt", 9],
                   "对称分牙下颌": ["resources/lower_9.pt", 9],
                   "齿龈分类": ["resources/齿龈分类.pt", 2],
                   "牙齿分区": ["resources/牙齿分区.pt", 3],
                   "磨尖牙分类": ["resources/磨尖牙分类.pt", 3],
                   "磨牙分类": ["resources/磨牙分类.pt", 3],
                   "切牙分类": ["resources/切牙分类.pt", 4],
                   "上颌": ["resources/upper_17.pt", 17],
                   "下颌": ["resources/lower_17.pt", 17], }


class generate_config:
    """
    生成配置，返回生成模型
    请求：
    file:文件
    192.168.1.X/generate

    """
    name = "generate"  # 主域名
    cache_dir = "resources/cache"  # 上传模型保存路径
    generate_model = "resources/全冠v1.1.1.tar"  # 上颌模型路径
    output_path = "resources/cache/generate_mesh.ply"  # 输出生成牙ply文件


class landmark_config:
    """
    标志点识别配置，返回三维标志点
    请求：
    file:文件
    192.168.1.X/landmark

    """
    name = "landmark"  # 主域名
    cache_dir = "resources/cache"  # 上传模型保存路径
    landmark_model = "resources/landmark.pt"  # 模型路径
    output_path = "resources/cache/landmark_mesh.ply"  # 输出调整后ply文件



class split_and_generate_config(split_config,generate_config):
    """
    通过分牙提取邻牙，并用全冠生成对应的牙位
    请求：
    file:文件
    constraints：优化强度
    refine_switch：优化开关
    model：选择运行模型（字符串，如"上颌"）
    id:1或2 （1取2，4 2取:13，15）
    地址：
    192.168.1.X/split_and_generate

    """
    name = "split_and_generate"  # 主域名

    ext_path = "resources/cache/ext.ply"  # 提取的邻牙信息
    miss_dental_path = "resources/cache/miss_dental.ply"  # 提取的缺失牙弓
    ori_miss_path = "resources/cache/miss_teeth.ply"  # 原始牙冠
    generate_path = "resources/cache/generate_teeth.ply"  # 生成的全冠

