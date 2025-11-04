import logging
import sys
import os
from datetime import datetime

def setup_logging(log_dir="./logs", log_filename=None, level=logging.INFO):
    """
    设置日志记录系统
    
    Args:
        log_dir: 日志文件目录
        log_filename: 日志文件名，如果为None则自动生成
        level: 日志级别
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"training_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录开始信息
    logging.info(f"Logging started. Log file: {log_path}")
    
    return log_path

class Logger:
    """简单的日志记录类"""
    
    def __init__(self, log_dir="./logs", name="training"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # 同时写入文件和控制台
        self.console = sys.stdout
        self.file = open(self.log_file, 'w', encoding='utf-8')
        
    def write(self, message):
        """写入消息到文件和控制台"""
        self.console.write(message)
        self.file.write(message)
        self.file.flush()  # 确保立即写入文件
        
    def flush(self):
        """刷新缓冲区"""
        self.console.flush()
        self.file.flush()
        
    def close(self):
        """关闭日志文件"""
        if self.file:
            self.file.close()
            
    def __enter__(self):
        """上下文管理器入口"""
        sys.stdout = self
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        sys.stdout = self.console
        self.close()

# 简单的日志函数
def log_to_file(message, log_dir="./logs", filename="training.log"):
    """快速记录消息到文件"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")