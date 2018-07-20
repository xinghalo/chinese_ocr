import yaml

with open('/Users/xingoo/PycharmProjects/chinese_ocr/ctpn/ctpn/text.yml', 'r', encoding='utf-8') as f:
    a = yaml.load(f)
    print(type(a))