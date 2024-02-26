import json

# JSON 파일 경로
json_file_path = "C:\group_project_data\makeData\\imageLabelCaptionTest.json"

# JSON 파일 열기
with open(json_file_path, "r") as f:
    data = json.load(f)

# 데이터 확인
print(data)

# keys = list(data.keys())
# print(keys)
# print(data[keys[0]])
# print(data[keys[0]]['captions'])