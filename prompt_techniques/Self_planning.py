import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class SelfplanGenerator(BaseGenerator):
    HumanEval_plan_prompt = '''
    Intent: Write a function to find the similar elements from the given two tuple lists.

    Plan: Let's think step by step.
    1. Define a function that takes two lists of tuples as arguments.
    2. Create an empty list to store the similar elements found in both tuple lists.
    3. Use a loop to iterate through each element (tuple) in the first list.
    4. For each tuple in the first list, check if it also exists in the second list.
    5. If the tuple from the first list is found in the second list, append it to the result list.
    6. After iterating through the first list, return the list containing the similar elements.

    Intent: Write a python function to identify non-prime numbers.

    Plan: Let's think step by step.
    1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
    2. Before proceeding to the prime check, consider edge cases.
    3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
    4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
    5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
    6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
    7. If no divisors are found, the function should return False (indicating the number is prime).

    Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

    Plan: Let's think step by step.
    1. Import the Required Module.
    2. Define a function that accepts two arguments.
    3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
    4. Use the heapq.nlargest Function.
    5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

    How about this intent: {prompt}.

    Plan: Let's think step by step.
    '''

    HumanEval_implementation_prompt = '''
    {prompt}
    Please complete the task with the following steps in Python.

    {plan}
    '''

    MBPP_plan_prompt = '''
    Intent: Write a function to find the similar elements from the given two tuple lists.

    Plan: Let's think step by step.
    1. Define a function that takes two lists of tuples as arguments.
    2. Create an empty list to store the similar elements found in both tuple lists.
    3. Use a loop to iterate through each element (tuple) in the first list.
    4. For each tuple in the first list, check if it also exists in the second list.
    5. If the tuple from the first list is found in the second list, append it to the result list.
    6. After iterating through the first list, return the list containing the similar elements.

    Intent: Write a python function to identify non-prime numbers.

    Plan: Let's think step by step.
    1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
    2. Before proceeding to the prime check, consider edge cases.
    3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
    4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
    5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
    6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
    7. If no divisors are found, the function should return False (indicating the number is prime).

    Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

    Plan: Let's think step by step.
    1. Import the Required Module.
    2. Define a function that accepts two arguments.
    3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
    4. Use the heapq.nlargest Function.
    5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

    How about this intent: {problem}.

    Plan: Let's think step by step.
    '''

    MBPP_implementation_prompt = '''
    {problem}
    Please complete the task with the following steps in Python.
    The function name and input variables should follow this template: {function_name}.

    {plan}
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
            return self.HumanEval_plan_prompt.format(prompt=prompt)
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_plan_prompt.format(problem=prompt)
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
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, function_name)}
                ]
            else:
                prompt = per_data.get('prompt', '')
                message = [
                    {'role': 'system', 'content': ''},
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
            total_input_token, total_output_token = 0, 0
            result = copy.copy(per_data)
            response1, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_output_token += output_token
            
            if 'HumanEval' in self.dataset_name:
                implement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.HumanEval_implementation_prompt.format(prompt=per_data['prompt'], plan=response1)}
                ]
            elif 'MBPP' in self.dataset_name:
                implement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.MBPP_implementation_prompt.format(problem=per_data['prompt'], function_name=utils.get_function_info(per_data['test_list'][0]), plan=response1)}
                ]
            
            response2, input_token, output_token = self.run_model(implement_message)
            total_input_token += input_token
            total_input_token += output_token
            code = utils.process_generation_to_code(response2)
            result['response_code'] = '\n'.join(code)
            result['input_token'] = total_input_token
            result['output_token'] = total_output_token
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
