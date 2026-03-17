import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer


# 使用参数初始化返回信息和端口
def run_server(port, message):
    class CustomHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(message.encode())

        def do_POST(self):
            self.do_GET()  # 所有请求统一响应

        def log_message(self, format, *args):
            return  # 可选：禁用日志输出

    server_address = ("", port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"Server listening on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple HTTP server")
    parser.add_argument("--port", type=int, required=True, help="Port to listen on")
    parser.add_argument("--message", type=str, required=True, help="Response message")
    args = parser.parse_args()

    run_server(args.port, args.message)
