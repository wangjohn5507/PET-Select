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

def APPS_check_code(final, input, output):
    signal.signal(signal.SIGALRM, timeout_handler)
    test = f'assert solution("{input.rstrip()}") == {output}'
    print(test)
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

def DS1000_check_code(final, test_code):
    exec(test_code, globals())
    try:
        exec(f'test_execution(\'{final}\')')
        print('Success')
        return True
    except TimeoutError as e:
        print('Test failed due to timeout:', str(e))
        return False
    except Exception as e:
        print(e)
        return False
    

def calculate_actual_acc(y_pred_classes, y_data_list, args):
    correct = 0
    zero_shot_correct = 0
    zero_cot_correct = 0
    few_shot_correct = 0
    few_cot_correct = 0
    self_correct = 0
    reflection_correct = 0
    total = 0
    for y_pred, y_data in zip(y_pred_classes, y_data_list):
        total += 1
        if y_pred == 0:
            strategy = f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 1:
            strategy = f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 2:
            strategy = f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 3:
            strategy = f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 4:
            strategy = f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 5:
            strategy = f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl'
        exec_record = y_data['exec_record']
        for exec in exec_record:
            if exec['strategy'] == strategy:
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    correct += 1
            if exec['strategy'] == f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    zero_shot_correct += 1
            if exec['strategy'] == f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    zero_cot_correct += 1
            if exec['strategy'] == f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    few_shot_correct += 1
            if exec['strategy'] == f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    few_cot_correct += 1
            if exec['strategy'] == f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    self_correct += 1
            if exec['strategy'] == f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl':
                exec_acc = exec['exec_acc']
                if exec_acc == 1:
                    reflection_correct += 1
    actual_acc = correct/total
    zero_shot_acc = zero_shot_correct/total
    zero_cot_acc = zero_cot_correct/total
    few_shot_acc =few_shot_correct/total
    few_cot_acc = few_cot_correct/total
    self_acc = self_correct/total
    reflection_acc = reflection_correct/total
    return actual_acc, zero_shot_acc, zero_cot_acc, few_shot_acc, few_cot_acc, self_acc, reflection_acc

def calculate_token_saved(y_pred_classes, y_data_list, args):
    pred_token = 0
    zero_shot_token = 0
    zero_cot_token = 0
    few_shot_token = 0
    few_cot_token = 0
    self_token = 0
    reflection_token = 0
    total = 0
    for y_pred, y_data in zip(y_pred_classes, y_data_list):
        total += 1
        if y_pred == 0:
            strategy = f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 1:
            strategy = f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 2:
            strategy = f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 3:
            strategy = f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 4:
            strategy = f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl'
        elif y_pred == 5:
            strategy = f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl'
        exec_record = y_data['exec_record']
        for exec in exec_record:
            if exec['strategy'] == strategy:
                token = exec['total_tokens']
                pred_token += token
            if exec['strategy'] == f'{args.dataset}_Zeroshot_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                zero_shot_token += token
            if exec['strategy'] == f'{args.dataset}_Zeroshot_CoT_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                zero_cot_token += token
            if exec['strategy'] == f'{args.dataset}_Fewshot_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                few_shot_token += token
            if exec['strategy'] == f'{args.dataset}_Fewshot_CoT_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                few_cot_token += token
            if exec['strategy'] == f'{args.dataset}_SelfDebug_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                self_token += token
            if exec['strategy'] == f'{args.dataset}_Reflection_{args.temperature}_{args.model}.jsonl':
                token = exec['total_tokens']
                reflection_token += token
    pred_avg_token = pred_token/total
    zero_shot_avg_token = zero_shot_token/total
    zero_cot_avg_token = zero_cot_token/total
    few_shot_avg_token = few_shot_token/total
    few_cot_avg_token = few_cot_token/total
    self_avg_token = self_token/total
    reflection_avg_token = reflection_token/total
    return pred_avg_token, zero_shot_avg_token, zero_cot_avg_token, few_shot_avg_token, few_cot_avg_token, self_avg_token, reflection_avg_token

