import argparse
import http.client


def run_client(ip, port):
    try:
        conn = http.client.HTTPConnection(ip, port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        print("Status:", response.status)
        print("Response:", response.read().decode())
        conn.close()
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple HTTP client")
    parser.add_argument("--ip", type=str, required=True, help="Server IP address")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    args = parser.parse_args()

    run_client(args.ip, args.port)
