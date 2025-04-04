import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model
from src import evaluation

from prompt_techniques.Techniques import BaseGenerator

class SelfdebugGenerator(BaseGenerator):
    HumanEval_SelfDebug_init_prompt = '''
    Complete the following task in Python:
    {prompt}

    Your code should pass the test:
    {test}
    '''

    MBPP_SelfDebug_init_prompt = '''
    {prompt}

    Your code should pass the test: 
    {test}

    The function name and input variables should follow this template: {function_name}.
    '''

    SelfDebug_success_prompt = '''
    {code}
    Is the code above correct? If not, please fix it.
    '''

    SelfDebug_failed_prompt = '''
    {code}
    The code above is wrong. Please fix it.
    '''

    APPS_SelfDebug_init_prompt = '''
    Complete the following task in Python:
    {prompt}

    Your code should pass the test:
    {test}
    '''

    def __init__(self, dataset_name, model_name, technique_name, args):
        """
        Initializes the ZeroShotGenerator with dataset, model, and additional arguments.
        """
        super().__init__(dataset_name, model_name, technique_name, args)

    def form_technique_prompt(self, prompt, test, function_name=None):
        """
        Forms the prompt string depending on the dataset_name. 
        """
        if 'HumanEval' in self.dataset_name:
            return self.HumanEval_SelfDebug_init_prompt.format(prompt=prompt, test=utils.get_first_elements_of_inputs_and_results(test))
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_SelfDebug_init_prompt.format(prompt=prompt, function_name=function_name, test=test)
        else:
            return self.APPS_SelfDebug_init_prompt.format(prompt=prompt, test=test)

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
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test'])}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test_list'][0], function_name)}
                ]
                print(message)
                # quit()
            else:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test'].split('\n')[3])}
                ]

            messages.append(message)
        return messages

    def run_model(self, message):
        if 'gpt' in self.model_name:
            return model.call_chat_gpt(message, self.args)
        else:
            return model.query_firework(message, self.args, self.model_name)

    def generate_result(self, messages, data, original_data):
        output_path = f'result/model_result/{self.dataset_name}_{self.technique_name}_{self.model_name}.jsonl'

        def run_func(message, per_data, per_original_data):
            tried = 0
            total_input_token, total_output_token = 0, 0
            result = copy.copy(per_data)
            response1, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_output_token += output_token
            code = utils.process_generation_to_code(response1)

            while(tried < 3):
                if 'HumanEval' in self.dataset_name:
                    one_assert = utils.extract_one_assert(per_original_data['test'])
                    passed = evaluation.check_code(per_data['prompt'], '\n'.join(code), f'def check(candidate):\n    {one_assert}\n', per_original_data['entry_point'])
                elif 'MBPP' in self.dataset_name:
                    one_assert = per_data['test_list'][0]
                    passed = evaluation.MBPP_check_code('\n'.join(code), per_data['test_list'])
                else:
                    one_assert = per_data['test'].split('\n')[3].strip().replace('candidate', 'solution')
                    passed = evaluation.check_apps('\n'.join(code), one_assert)
                if passed:
                    debug_message = [
                        {'role': 'system', 'content': self.system_message},
                        {'role': 'user', 'content': self.SelfDebug_success_prompt.format(code='\n'.join(code))}
                    ]
                    response2, input_token, output_token = self.run_model(debug_message)
                    total_input_token += input_token
                    total_output_token += output_token
                    code = utils.process_generation_to_code(response2)
                    break
                else:
                    debug_message = [
                        {'role': 'system', 'content': self.system_message},
                        {'role': 'user', 'content': self.SelfDebug_failed_prompt.format(code='\n'.join(code))}
                    ]
                    response2, input_token, output_token = self.run_model(debug_message)
                    total_input_token += input_token
                    total_output_token += output_token
                    code = utils.process_generation_to_code(response2)
                    tried += 1
    
            result['response_code'] = '\n'.join(code)
            result['input_token'] = total_input_token
            result['output_token'] = total_output_token
            return result

        responses = []

        # Run generation concurrently
        with cfuts.ThreadPoolExecutor(max_workers=32) as executor:
            futs = []
            for idx, per_data in enumerate(data):
                futs.append(executor.submit(run_func, messages[idx], per_data, original_data[idx]))

            for future in tqdm(cfuts.as_completed(futs), total=len(futs)):
                responses.append(future.result())

        # Sort results by task_id if it exists in your dataset
        responses.sort(key=lambda x: int(x['task_id']))

        # Write out to a JSON lines file
        with open(output_path, 'a') as f:
            for res in responses:
                f.write(json.dumps(res) + "\n")
