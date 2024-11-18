import json
import os

attack_data_path = './attack_data'
normal_data_path = './normal_data'


def load_data_from_folder(folder_path, save_dir):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                transaction_data = json.load(f)

                if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
                    data_map = transaction_data['trace']['dataMap']
                    cleaned_data = clean_data(data_map)

                    save_to_txt(filename.replace('.json', '.txt'), cleaned_data, save_dir)


def clean_data(data_map):
    cleaned_lines = []

    for key, value in data_map.items():
        node_type = value.get('nodeType')
        if node_type == 0 and 'invocation' in value:
            invocation = value['invocation']
            operation = invocation.get('operation', 'N/A')
            if operation == 'CALL' or operation == 'DELEGATECALL':
                id = invocation.get('id')
                call_data = invocation.get('callData', 'N/A')
                call_params = invocation.get('callParams', [])
                decoded_method = invocation.get('decodedMethod', {})
                gas_used = invocation.get('gasUsed', 0)
                selector = invocation.get('selector', 'N/A')
                value = invocation.get('value', 0)

                cleaned_line = f"{id},{operation},{call_data},{call_params},{decoded_method},{gas_used},{selector},{value}"
                cleaned_lines.append(cleaned_line)

    return '\n'.join(cleaned_lines)


def save_to_txt(filename, content, save_dir):
    with open(os.path.join(save_dir, filename), 'w', encoding='utf-8') as f:
        f.write(content)


load_data_from_folder(attack_data_path, './processed_attack_data')
load_data_from_folder(normal_data_path, './processed_normal_data')
