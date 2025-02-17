import ast
import textwrap
import signal   
import subprocess

def timeout_handler(signum, frame):
    raise TimeoutError("Test execution exceeded time limit")

def extract_function_body(code, entry_point):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                code = ast.unparse(node.body)
                indent_str = '    '
                indented_code = textwrap.indent(text=code, prefix=indent_str)
                return indented_code
    except:
        return code
    
def check_code(prompt, final, test, entry_point):
    signal.signal(signal.SIGALRM, timeout_handler)
    final = extract_function_body(final, entry_point)
    if final != None:
        final_code = prompt + final
    else:
        final_code = prompt
    
    try:
        exec(final_code)
        print(final_code)
    except:
        print('wrong code')
        return False
    
    signal.alarm(10)
    exec(test)

    try:
        locals()['check']((locals()[entry_point]))
        print('Success')
        return True
    except Exception as e:
        # print(e)
        return False
    finally:
        signal.alarm(0)  # Cancel the alarm
    
def MBPP_check_code(final, test):
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        exec(final)
        print(final)
    except:
        print('wrong code')
        return False
    
    signal.alarm(10)
    try:
        exec(test)
        print('Success')
        return True
    except TimeoutError as e:
        print('Test failed due to timeout:', str(e))
        return False
    except Exception as e:
        return False
    finally:
        signal.alarm(0)  # Cancel the alarm

def eval_humaneval(prompt, code, test, entry_point):
    if entry_point not in code:
        code = prompt + code
    test = test.replace('candidate', entry_point)
    full_test = '''
{code}

{test}

check({entry_point})
    '''

    full_test = full_test.format(code=code, test=test, entry_point=entry_point)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        # print(full_test)
        # print(e)
        print("failed")
        return False
    
def eval_mbpp(code, test_string, is_plus):
    if not is_plus:
        full_test = '''
{code}

test_list = {test_string}

def run_tests():
    """
    Executes each test in test_list using 'exec'.
    If all assertions pass, it prints a success message.
    """
    for test in test_list:
        # Execute each test string, which includes the assert statement
        exec(test)

if __name__ == "__main__":
    run_tests()
        '''
    else:
        full_test = '''
{code}

{test_string}
'''
        
    full_test = full_test.format(code=code, test_string=test_string)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    # print(full_test)
    # quit()
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        print("failed")
        return False

    

def eval_apps(code, test_string):
    full_test = '''
{code}

{test_string}

check(solution)
'''
    full_test = full_test.format(code=code, test_string=test_string)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    # print(full_test)
    # quit()
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        print("failed")
        return False
    
def check_apps(code, assertion):
    full_test = '''
{code}

{assertion}
'''
    full_test = full_test.format(code=code, assertion=assertion)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    # print(full_test)
    # quit()
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        print("failed")
        return False