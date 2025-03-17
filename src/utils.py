# import tiktoken
import json
import ast
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from radon.complexity import cc_visit
from radon.visitors import ComplexityVisitor
from radon.metrics import h_visit
from radon.metrics import mi_visit
from cognitive_complexity.api import get_cognitive_complexity


def process_generation_to_code(gens):
    if '```python' in gens:
        gens = gens.split('```python')[1].split('```')[0]
    elif '```' in gens:
        gens = gens.split('```')[1].split('```')[0]
        
    return gens.split('\n')[1:-1]


def write_to_file(result, output_file):
    # print(output_file)
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

def extract_one_assert(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Define a function to find all assert statements in the code
    def find_asserts(node):
        results = []
        for n in ast.walk(node):
            if isinstance(n, ast.Assert):
                results.append(n)
        return results

    # Get all assert statements
    asserts = find_asserts(tree)

    # Print the first assert statement
    if asserts:
        first_assert = asserts[0]
        return ast.unparse(first_assert)
    else:
        print("No assert statements found.")

def get_first_elements_of_inputs_and_results(code_string):
    tree = ast.parse(code_string)
    inputs_first_val = None
    results_first_val = None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check if this assignment is to 'inputs' or 'results'
                    if target.id == 'inputs':
                        # If this is a list literal (ast.List), extract the first element if present
                        if isinstance(node.value, ast.List) and node.value.elts:
                            inputs_first_val = ast.literal_eval(node.value.elts[0])
                    elif target.id == 'results':
                        if isinstance(node.value, ast.List) and node.value.elts:
                            results_first_val = ast.literal_eval(node.value.elts[0])

    return inputs_first_val, results_first_val

def extract_function_name_from_assert(test_cases):
    function_names = set()
    
    for test_case in test_cases:
        tree = ast.parse(test_case)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                test = node.test
                if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
                    function_name = test.left.func.id
                    function_names.add(function_name)
    
    return list(function_names)[0]

def get_function_info(code):
    call_strings = code.split('assert ')[1]
                    
    return 'def '+ call_strings

def get_largest_and_smallest_physical_loc(dataset, is_humaneval=True, is_apps=True):
    largest = 0
    smallest = 100
    for idx, per_data in enumerate(dataset):
        if is_humaneval:
            code = per_data['ground_truth_code']
            code = 'def function():\n' + code
        elif is_apps==False:
            code = per_data['ground_truth_code']
        else:
            if idx == 100 or idx == 455 or idx == 499 or idx == 2888 or idx == 2924:
                code = per_data['ground_truth_code_list'][1]
            elif idx == 559 or idx == 1431 or idx==2653 or idx==2661 or idx==2721:
                code = per_data['ground_truth_code_list'][2]
            elif idx == 2723:
                code = per_data['ground_truth_code_list'][10]
            elif idx == 2598 or idx == 2726 or idx == 3763:
                continue
            else:
                code = per_data['ground_truth_code']
        largest = max(largest, count_physical_loc(code))
        smallest = min(smallest, count_physical_loc(code))
    return largest, smallest

def get_largest_and_smallest_cyclomatic_complexity(dataset, is_humaneval=True, is_apps=True):
    largest = 0
    smallest = 100
    for idx, per_data in enumerate(dataset):
        # print(idx)
        if is_humaneval:
            code = per_data['ground_truth_code']
            code = 'def function():\n' + code
        elif is_apps==False:
            code = per_data['ground_truth_code']
        else:
            if idx == 100 or idx == 455 or idx == 499 or idx == 2888 or idx == 2924:
                code = per_data['ground_truth_code_list'][1]
            elif idx == 559 or idx == 1431 or idx==2653 or idx==2661 or idx==2721:
                code = per_data['ground_truth_code_list'][2]
            elif idx == 2723:
                code = per_data['ground_truth_code_list'][10]
            elif idx == 2598 or idx == 2726 or idx == 3763:
                continue
            else:
                code = per_data['ground_truth_code']
        # print(idx)
        largest = max(largest, calculate_cyclomatic_complexity(code))
        smallest = min(smallest, calculate_cyclomatic_complexity(code))
    return largest, smallest

def get_largest_and_smallest_halstead_complexity(dataset, is_humaneval=True, is_apps=True):
    largest = 0
    smallest = 100
    for idx, per_data in enumerate(dataset):
        if is_humaneval:
            code = per_data['ground_truth_code']
            code = 'def function():\n' + code
        elif is_apps==False:
            code = per_data['ground_truth_code']
        else:
            if idx == 100 or idx == 455 or idx == 499 or idx == 2888 or idx == 2924:
                code = per_data['ground_truth_code_list'][1]
            elif idx == 559 or idx == 1431 or idx==2653 or idx==2661 or idx==2721:
                code = per_data['ground_truth_code_list'][2]
            elif idx == 2723:
                code = per_data['ground_truth_code_list'][10]
            elif idx == 2598 or idx == 2726 or idx == 3763:
                continue
            else:
                code = per_data['ground_truth_code']
        largest = max(largest, calculate_halstead_complexity(code))
        smallest = min(smallest, calculate_halstead_complexity(code))
    return largest, smallest

def get_largest_and_smallest_mi(dataset, is_humaneval=True, is_apps=True):
    largest = 0
    smallest = 100
    for idx, per_data in enumerate(dataset):
        if is_humaneval:
            code = per_data['ground_truth_code']
            code = 'def function():\n' + code
        elif is_apps==False:
            code = per_data['ground_truth_code']
        else:
            if idx == 100 or idx == 455 or idx == 499 or idx == 2888 or idx == 2924:
                code = per_data['ground_truth_code_list'][1]
            elif idx == 559 or idx == 1431 or idx==2653 or idx==2661 or idx==2721:
                code = per_data['ground_truth_code_list'][2]
            elif idx == 2723:
                code = per_data['ground_truth_code_list'][10]
            elif idx == 2598 or idx == 2726 or idx == 3763:
                continue
            else:
                code = per_data['ground_truth_code']
        largest = max(largest, calculate_mi(code))
        smallest = min(smallest, calculate_mi(code))
    return largest, smallest

def get_largest_and_smallest_cognitive_complexity(dataset, is_humaneval=True, is_apps=True):
    largest = 0
    smallest = 100
    for idx, per_data in enumerate(dataset):
        if is_humaneval:
            code = per_data['ground_truth_code']
            code = 'def function():\n' + code
        elif is_apps==False:
            code = per_data['ground_truth_code']
        else:
            if idx == 100 or idx == 455 or idx == 499 or idx == 2888 or idx == 2924:
                code = per_data['ground_truth_code_list'][1]
            elif idx == 559 or idx == 1431 or idx==2653 or idx==2661 or idx==2721:
                code = per_data['ground_truth_code_list'][2]
            elif idx == 2723:
                code = per_data['ground_truth_code_list'][10]
            elif idx == 2598 or idx == 2726 or idx == 3763:
                continue
            else:
                code = per_data['ground_truth_code']
        largest = max(largest, calculate_cognitive_complexity(code))
        smallest = min(smallest, calculate_cognitive_complexity(code))
    return largest, smallest

def calculate_weighted_complexity(code, largest_physical_loc, smallest_physical_loc, largest_cyclomatic_complexity, smallest_cyclomatic_complexity, largest_halstead_complexity, smallest_halstead_complexity, largest_mi, smallest_mi, largest_cognitive_complexity, smallest_cognitive_complexity):

    physical_loc = count_physical_loc(code)
    cyclomatic_complexity = calculate_cyclomatic_complexity(code)
    halstead_complexity = calculate_halstead_complexity(code)
    mi = calculate_mi(code)
    cognitive_complexity = calculate_cognitive_complexity(code)

    # largest_physical_loc, smallest_physical_loc = get_largest_and_smallest_physical_loc(dataset, is_humaneval, is_apps)
    # largest_cyclomatic_complexity, smallest_cyclomatic_complexity = get_largest_and_smallest_cyclomatic_complexity(dataset, is_humaneval, is_apps)
    # largest_halstead_complexity, smallest_halstead_complexity = get_largest_and_smallest_halstead_complexity(dataset, is_humaneval, is_apps)
    # largest_mi, smallest_mi = get_largest_and_smallest_mi(dataset, is_humaneval, is_apps)
    # largest_cognitive_complexity, smallest_cognitive_complexity = get_largest_and_smallest_cognitive_complexity(dataset, is_humaneval, is_apps)

    normalized_physical_loc = (physical_loc - smallest_physical_loc) / (largest_physical_loc - smallest_physical_loc)
    normalized_cyclomatic_complexity = (cyclomatic_complexity - smallest_cyclomatic_complexity) / (largest_cyclomatic_complexity - smallest_cyclomatic_complexity)
    normalized_halstead_complexity = (halstead_complexity - smallest_halstead_complexity) / (largest_halstead_complexity - smallest_halstead_complexity)
    normalized_mi = (mi - smallest_mi) / (largest_mi - smallest_mi)
    normalized_cognitive_complexity = (cognitive_complexity - smallest_cognitive_complexity) / (largest_cognitive_complexity - smallest_cognitive_complexity)

    # print(normalized_physical_loc, normalized_cyclomatic_complexity, normalized_halstead_complexity, normalized_mi, normalized_cognitive_complexity)
    
    # Define the weights for each complexity metric
    # weights = {
    #     'physical_loc': 0.2,
    #     'cyclomatic_complexity': 0.5,
    #     'halstead_complexity': 0.05,
    #     'mi': 0.05,
    #     'cognitive_complexity': 0.2
    # }
    weights = {
        'physical_loc': 0.1,
        'cyclomatic_complexity': 0.2,
        'halstead_complexity': 0.15,
        'mi': 0.25,
        'cognitive_complexity': 0.3
    }
    
    # Calculate the weighted complexity
    weighted_complexity = (
        weights['physical_loc'] * normalized_physical_loc +
        weights['cyclomatic_complexity'] * normalized_cyclomatic_complexity +
        weights['halstead_complexity'] * normalized_halstead_complexity +
        weights['mi'] * normalized_mi +
        weights['cognitive_complexity'] * normalized_cognitive_complexity
    )
    
    return weighted_complexity, normalized_physical_loc * 100, normalized_cyclomatic_complexity * 100, normalized_halstead_complexity * 100, normalized_mi * 100, normalized_cognitive_complexity * 100



def count_physical_loc(code_string):
    # Split the input string into lines
    lines = code_string.split('\n')
    
    # Filter out empty lines and count the remaining lines
    non_empty_lines = [line for line in lines if line.strip() != '' and not line.strip().startswith('#')]
    
    return len(non_empty_lines)

def calculate_cyclomatic_complexity(code):
    # Analyze the code
    # print('Code:' + code)
    blocks = cc_visit(code)
    # for block in blocks:
    #     print(f'{block.name}: {block.complexity}')

    # Calculate the average Cyclomatic Complexity
    total_complexity = sum(block.complexity for block in blocks)
    average_complexity = total_complexity / len(blocks) if blocks else 0
    # print(f'Average Cyclomatic Complexity: {average_complexity}')
    return average_complexity

def calculate_halstead_complexity(code):
    results = h_visit(code)
    return results[0].vocabulary

def calculate_mi(code_string):
    mi_score = mi_visit(code_string, True)
    return 100-mi_score

def wrap_top_level_in_function(code_str: str) -> str:
    """
    Wrap all code in a synthetic function _top_level_.
    This allows python-cognitive-complexity to analyze even the top-level statements.
    """
    # Indent every line (except possibly empty lines)
    wrapped_lines = ["def _top_level_():", ""]
    for line in code_str.splitlines():
        if line.strip():  # non-empty
            wrapped_lines.append("    " + line)
        # else:
        #     wrapped_lines.append("    pass")  # or just append blank lines
    wrapped_lines.append("_top_level_()")  # call the synthetic function at the end
    return "\n".join(wrapped_lines)

def calculate_cognitive_complexity(code):
    # print(code)
    code = wrap_top_level_in_function(code)
    # print(code)
    parsed_code = ast.parse(code)
    try:
        new_body = [node for node in parsed_code.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.For, ast.AugAssign, ast.If))]
        if not new_body:
            funcdef = ast.parse('')

        else:
            funcdef = new_body[0]
            
    except Exception as e:
        print(e)
        print('Using original code.')
        if not parsed_code.body:
            raise ValueError("The code provided is empty or invalid.")
        funcdef = parsed_code.body[0]
    
    cc_score = get_cognitive_complexity(funcdef)
   
    return cc_score

def extract_exec_code(code_str):
    """
    Extracts the value of the exec_context variable from the given code string.
    
    Parameters:
    code_str (str): A string containing the Python code.
    
    Returns:
    str: The value of the exec_context variable, or None if not found.
    """
    # Parse the code string into an abstract syntax tree (AST)
    tree = ast.parse(code_str)

    # Define a visitor class to extract the exec_context variable
    class ExecContextExtractor(ast.NodeVisitor):
        def __init__(self):
            self.exec_context = None

        def visit_Assign(self, node):
            # Check if the variable being assigned is exec_context
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'exec_context':
                # Extract the value of exec_context
                self.exec_context = ast.literal_eval(node.value)
    
    # Create an instance of the visitor class and visit the tree
    extractor = ExecContextExtractor()
    extractor.visit(tree)
    
    # Return the extracted exec_context value as a string
    if extractor.exec_context:
        return extractor.exec_context
    else:
        return None