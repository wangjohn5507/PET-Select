import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class FewshotCoTGenerator(BaseGenerator):
    HumanEval_Fewshot_prompt = '''
    Here are some examples of how to generate the code step by step.

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
        Let's complete the following code step by step.
        """
        # Step 1: Create a variable to store the result
        result = False
        # Step 2: Loop through the list of numbers
        for i in range(len(numbers)):
            # Step 3: Check if the current number is within the threshold of any other number in the list
            for j in range(i+1, len(numbers)):
                if abs(numbers[i] - numbers[j]) <= threshold:
                    # Step 4: If the condition is met, set the result to True and break out of the loop
                    result = True
                    break
            # Step 5: If the result is already True, break out of the loop
            if result:
                break

        # Step 6: Return the result
        return result
    ```

    Example 2:

    ```python
    from typing import List
    def rescale_to_unit(numbers: List[float]) -> List[float]:
        """ Given list of numbers (of at least two elements), apply a linear transform to that list,
        such that the smallest number will become 0 and the largest will become 1
        >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
        [0.0, 0.25, 0.5, 0.75, 1.0]
        Let's complete the following code step by step.
        """
        # Step 1: Find the smallest and largest numbers in the list
        smallest = min(numbers)
        largest = max(numbers)
        # Step 2: Calculate the difference between the largest and smallest numbers
        difference = largest - smallest
        # Step 3: Create a new list to store the rescaled numbers
        rescaled_numbers = []
        # Step 4: Loop through each number in the original list
        for number in numbers:
            # Step 5: Apply the linear transform to each number
            rescaled_number = (number - smallest) / difference
            # Step 6: Add the rescaled number to the new list
            rescaled_numbers.append(rescaled_number)
        # Step 7: Return the new list
        return rescaled_numbers
    ```

    Example 3:

    ```python
    def strlen(string: str) -> int:
        """ Return length of given string
        >>> strlen('')
        0
        >>> strlen('abc')
        3
        Let's complete the following code step by step.
        """
        # 1. Initialize a variable to store the length of the string
        length = 0
        # 2. Use a for loop to iterate through each character in the string
        for char in string:
            # 3. Increment the length variable by 1 for each character
            length += 1
        # 4. Return the length variable
        return length
    ```

    How about this function?
    {prompt}
    '''

    MBPP_Fewshot_CoT_prompt = '''
    You are an expert Python programmer, and you should write code step by step to complete the task.

    Example 1:
    Here is your task: Write a function to find the similar elements from the given two tuple lists.
    Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

    ```python
    def similar_elements(test_tup1, test_tup2):
        # Convert both tuples to sets to remove duplicates and allow for set operations
        res = tuple(set(test_tup1) & set(test_tup2))
        # The '&' operator finds the intersection of the two sets, i.e., elements common to both sets
        return (res)  # Return the common elements as a tuple
    ```

    Example 2:
    Here is your task: Write a python function to identify non-prime numbers.
    Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

    ```python
    import math  # Import the math module to use mathematical functions

    def is_not_prime(n):
        result = False  # Initialize a variable 'result' to False. It indicates whether 'n' is not a prime number.

        # Loop from 2 to the square root of 'n', rounded down to the nearest whole number, then add 1 to include that number in the loop.
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:  # Check if 'n' is divisible by 'i' (i.e., no remainder)
                result = True  # If 'n' is divisible by any number other than 1 and itself, set 'result' to True.

        return result  # Return the value of 'result'. True if 'n' is not a prime number, False if 'n' is a prime number.
    ```

    Example 3:
    Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
    Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

    ```python
    import heapq as hq  # Import the heapq module and rename it as hq for convenience

    def heap_queue_largest(nums, n):
        # Use the heapq.nlargest function to get the 'n' largest elements from the list 'nums'
        largest_nums = hq.nlargest(n, nums)
        return largest_nums  # Return the list of the 'n' largest elements
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
            return self.MBPP_Fewshot_CoT_prompt.format(prompt=prompt, function_name=function_name)
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
