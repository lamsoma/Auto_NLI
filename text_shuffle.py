import random

def shuffle_text_file(input_file_path, output_file_path):
    # ファイルを読み込んで各行をリストに格納
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # リスト内の行をシャッフル
    random.shuffle(lines)

    # シャッフルされた内容を新しいファイルに書き出し
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

# 使用例
input_file_path = '/home/sato/ja/wiki_40b_train_en.txt'  # 入力ファイルのパス
output_file_path = '/home/sato/ja/wiki_40b_train_en_shuffle.txt'  # 出力ファイルのパス

shuffle_text_file(input_file_path, output_file_path)
