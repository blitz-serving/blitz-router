import threading

def _wait_for_output(process, keyword, on_ready):
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8', errors='ignore').strip()
        print("Child:", decoded_line)  # 可选：打印或记录日志
        if keyword in decoded_line:
            on_ready()
            break
        
def block_until_keyword(proc, keyword):
    ready_flag = threading.Event()

    thread = threading.Thread(
        target=_wait_for_output,
        args=(proc, keyword, ready_flag.set),
        daemon=True
    )
    thread.start()
    ready_flag.wait()