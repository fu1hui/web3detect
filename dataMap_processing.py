import json
import os

attack_data_path = './attack_data'
normal_data_path = './normal_data'


def load_data_from_folder(folder_path, save_dir):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
                    transaction_data = json.load(f)

                    fc_num = function_count(transaction_data)
                    gas = get_gas_value(transaction_data)



def function_count(transaction_data):
    max_function_num = []
    function_dic = {}

    # 检查 'trace' 和 'dataMap' 是否存在
    if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
        data_map = transaction_data['trace']['dataMap']

        for key, value in data_map.items():
            node_type = value.get('nodeType')
            function = None

            if node_type == 0:
                function = value.get('invocation', {}).get('decodedMethod', {}).get('name')
            elif node_type == 1:
                function = value.get('event', {}).get('decodedLog', {}).get('name')

            if function:
                function_dic[function] = function_dic.get(function, 0) + 1

        # 获取最大值并添加到列表中
        max_value = max(function_dic.values(), default=-1)
        max_function_num.append(max_value)

    return max_function_num


def get_gas_value(transaction_data):
    profile = transaction_data.get('profile', {})
    basic_info = profile.get('basic_info', {})
    if 'gasUsed' in basic_info and basic_info['gasUsed']:
        return basic_info['gasUsed']
    return -1


load_data_from_folder(attack_data_path)
load_data_from_folder(normal_data_path)
