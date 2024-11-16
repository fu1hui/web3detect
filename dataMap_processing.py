import json
import os

attack_data_path = './attack_data'
normal_data_path = './normal_data'


#####################################
## 保存文件还没有写，先把你的数据处理写上 ##
#####################################

def load_data_from_folder(folder_path, list1, list2):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                transaction_data = json.load(f)
                if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
                    fc_num = function_count(transaction_data)
                    gas = get_gas_value(transaction_data)
                    if filename == '0x48164d3adbab78c2cb9876f6e17f88e321097fcd14cadd57556866e4ef3e185d.json' :
                        print(fc_num, gas)  # 测试
                    list1.append(fc_num)
                    list2.append(gas)


def function_count(transaction_data):
    function_dic = {}

    if 'trace' in transaction_data and 'dataMap' in transaction_data['trace']:
        data_map = transaction_data['trace']['dataMap']

        for key, value in data_map.items():
            node_type = value.get('nodeType')
            function = None

            if node_type == 0:
                invocation = value.get('invocation')
                if isinstance(invocation, dict):
                    decoded_method = invocation.get('decodedMethod')
                    if isinstance(decoded_method, dict):
                        function = decoded_method.get('name')
            elif node_type == 1:
                event = value.get('event')
                if isinstance(event, dict):
                    decoded_log = event.get('decodedLog')
                    if isinstance(decoded_log, dict):
                        function = decoded_log.get('name')

            if function:
                function_dic[function] = function_dic.get(function, 0) + 1

    max_value = max(function_dic.values(), default=-1)

    return max_value


def get_gas_value(transaction_data):
    profile = transaction_data.get('profile', {})
    basic_info = profile.get('basic_info', {})
    if 'gasUsed' in basic_info and basic_info['gasUsed']:
        return basic_info['gasUsed']
    return -1


normal_fc_num = []
attack_fc_num = []
normal_gas = []
attack_gas = []
load_data_from_folder(attack_data_path, attack_fc_num, attack_gas)
load_data_from_folder(normal_data_path, normal_fc_num, normal_gas)
print(attack_fc_num, '\n', normal_fc_num, '\n', attack_gas, '\n', normal_gas)
