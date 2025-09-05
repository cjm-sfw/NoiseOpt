from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import logging
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("frontend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保日志消息立即刷新到文件
logging.getLogger().handlers[0].flush = lambda: logging.getLogger().handlers[0].stream.flush()

# 加载环境变量
load_dotenv()

# 获取前端配置
host = os.getenv("FRONTEND_HOST", "0.0.0.0")
port = int(os.getenv("FRONTEND_PORT", 3000))

# 设置当前目录为前端文件目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

if __name__ == "__main__":
    server_address = (host, port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    logger.info(f"Frontend server running on http://{host}:{port}")
    print(f"Frontend server running on http://{host}:{port}")
    httpd.serve_forever()
