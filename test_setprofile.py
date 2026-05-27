import sys

def test_func():
    stats_visits = {1: 100, 2: 200}
    return 5

def trace(frame, event, arg):
    if event == 'return' and frame.f_code.co_name == 'test_func':
        print("locals:", frame.f_locals)
    return trace

sys.setprofile(trace)
test_func()
sys.setprofile(None)
