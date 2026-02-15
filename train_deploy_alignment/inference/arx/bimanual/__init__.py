import os
import sys

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

def find_first_specific_so_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件名是否以 'arx_l5pro_python_api' 开头，并且以 '.so' 结尾
            if filename.startswith('arx_x5_python') and filename.endswith('.so'):
                # 返回第一个找到的符合条件的文件路径
                return os.path.join(dirpath, filename)
    return None  # 如果没有找到符合条件的文件，则返回 None

# 使用 os.path.join 来拼接路径
so_file = find_first_specific_so_file(os.path.join(current_dir, 'api', 'arx_x5_python'))

# 确保共享库的路径在 Python 的路径中
if os.path.exists(so_file):
    sys.path.append(os.path.dirname(so_file))  # 添加共享库所在目录到 sys.path
else:
    raise FileNotFoundError(f"Shared library not found: {so_file}")

# 导入 Python 模块
try:
    from .script.dual_arm import BimanualArm  # 确保这两个类在各自的文件中被正确定义
    from .script.single_arm import SingleArm
except ImportError as e:
    raise ImportError(f"Failed to import Python modules: {e}")

# 可选：定义 __all__ 以控制模块导出的内容
__all__ = ['BimanualArm', 'SingleArm', 'arx_x5_python']
