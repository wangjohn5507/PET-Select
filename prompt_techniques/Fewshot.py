import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class FewshotGenerator(BaseGenerator):
    HumanEval_Fewshot_prompt = '''
    There are some examples of how to generate the code.

    Example 1:

    ```python
    from typing import List
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        """ Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        """
        for idx, elem in enumerate(numbers):
            for idx2, elem2 in enumerate(numbers):
                if idx != idx2:
                    distance = abs(elem - elem2)
                    if distance < threshold:
                        return True

        return False
    ```

    Example 2:

    ```python
    from typing import List
    def separate_paren_groups(paren_string: str) -> List[str]:
        """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
        separate those group into separate strings and return the list of those.
        Separate groups are balanced (each open brace is properly closed) and not nested within each other
        Ignore any spaces in the input string.
        >>> separate_paren_groups('( ) (( )) (( )( ))')
        ['()', '(())', '(()())']
        """
        result = []
        current_string = []
        current_depth = 0

        for c in paren_string:
            if c == '(':
                current_depth += 1
                current_string.append(c)
            elif c == ')':
                current_depth -= 1
                current_string.append(c)

                if current_depth == 0:
                    result.append(''.join(current_string))
                    current_string.clear()

        return result
    ```

    Example 3:

    ```python
    def truncate_number(number: float) -> float:
        """ Given a positive floating point number, it can be decomposed into
        and integer part (largest integer smaller than given number) and decimals
        (leftover part always smaller than 1).

        Return the decimal part of the number.
        >>> truncate_number(3.5)
        0.5
        """
        return number % 1.0
    ```

    How about this function?
    {prompt}
    '''

    MBPP_Fewshot_prompt = '''
    Here are some examples of how to generate the code.

    Example 1:
    Here is your task: Write a function to find the similar elements from the given two tuple lists.
    Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

    ```python
    def similar_elements(test_tup1, test_tup2):
        res = tuple(set(test_tup1) & set(test_tup2))
        return (res) 
    ```

    Example 2:
    Here is your task: Write a python function to identify non-prime numbers.
    Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

    ```python
    import math
    def is_not_prime(n):
        result = False
        for i in range(2,int(math.sqrt(n)) + 1):
            if n % i == 0:
                result = True
        return result
    ```

    Example 3:
    Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
    Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

    ```python
    import heapq as hq
    def heap_queue_largest(nums,n):
        largest_nums = hq.nlargest(n, nums)
        return largest_nums
    ```

    How about this task?
    Here is your task: {prompt}
    The function name and input variables should follow this template: {function_name}.
    '''

    def __init__(self, dataset_name, model_name, technique_name, args):
        """
        Initializes the ZeroShotGenerator with dataset, model, and additional arguments.
        """
        super().__init__(dataset_name, model_name, technique_name, args)

    def form_technique_prompt(self, prompt, function_name=None):
        """
        Forms the prompt string depending on the dataset_name. 
        """
        if 'HumanEval' in self.dataset_name:
            return self.HumanEval_Fewshot_prompt.format(prompt=prompt)
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_Fewshot_prompt.format(prompt=prompt, function_name=function_name)
        else:
            return prompt

    def generate_prompt(self, dataset):
        """
        Generates the list of messages for each data item in the dataset.
        """
        messages = []
        for per_data in dataset:
            # Check dataset type
            if 'HumanEval' in self.dataset_name:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, function_name)}
                ]
            else:
                prompt = per_data.get('prompt', '')
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
                ]

            messages.append(message)
        return messages

    def run_model(self, message):
        if 'gpt' in self.model_name:
            return model.call_chat_gpt(message, self.args)
        else:
            return model.query_firework(message, self.args, self.model_name)

    def generate_result(self, messages, data):
        output_path = f'result/model_result/{self.dataset_name}_{self.technique_name}_{self.model_name}.jsonl'

        def run_func(message, per_data):
            result = copy.copy(per_data)
            response, input_token, output_token = self.run_model(message)
            code = utils.process_generation_to_code(response)
            result['response_code'] = '\n'.join(code)
            result['input_token'] = input_token
            result['output_token'] = output_token
            return result

        responses = []

        # Run generation concurrently
        with cfuts.ThreadPoolExecutor(max_workers=32) as executor:
            futs = []
            for idx, per_data in enumerate(data):
                futs.append(executor.submit(run_func, messages[idx], per_data))

            for future in tqdm(cfuts.as_completed(futs), total=len(futs)):
                responses.append(future.result())

        # Sort results by task_id if it exists in your dataset
        responses.sort(key=lambda x: int(x['task_id']))

        # Write out to a JSON lines file
        with open(output_path, 'w') as f:
            for res in responses:
                f.write(json.dumps(res) + "\n")
