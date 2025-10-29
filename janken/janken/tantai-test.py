"""
tantai-test-img内の画像をすべて推測して結果をMarkdownレポートにまとめる
フォルダ分けされていない画像を個別に判定
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

# 設定
MODEL_PATH = "model_with_subdirs.keras"
IMAGE_DIR = "tantai-test-img"
OUTPUT_MD = "tantai_test_result.md"
TARGET_SIZE = (224, 224)

# クラス名
CLASS_NAMES = {
    0: "ぐー",
    1: "ちょき", 
    2: "ぱー"
}

def load_model():
    """学習済みモデルをロード"""
    print("🔄 モデルをロード中...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ モデルロード完了")
    return model

def preprocess_image(image_path):
    """画像を前処理"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_images(model, image_dir):
    """
    指定ディレクトリ内のすべての画像を推測
    
    Returns:
        results: [(filename, predicted_class, confidence), ...]
    """
    results = []
    
    # 画像ファイルを取得（重複を防ぐためsetを使用）
    image_files = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG', '*.bmp', '*.BMP']:
        image_files.update(Path(image_dir).glob(ext))
    
    image_files = sorted(list(image_files))
    
    if not image_files:
        print(f"⚠️ {image_dir} に画像が見つかりません")
        return results
    
    print(f"\n📊 {len(image_files)}枚の画像を処理中...")
    
    for i, image_path in enumerate(image_files, 1):
        try:
            # 画像を前処理
            img_array = preprocess_image(image_path)
            
            # 推測
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            results.append({
                'filename': image_path.name,
                'predicted_class': CLASS_NAMES[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'ぐー': predictions[0][0],
                    'ちょき': predictions[0][1],
                    'ぱー': predictions[0][2]
                }
            })
            
            # 進捗表示
            if i % 10 == 0 or i == len(image_files):
                print(f"  処理中: {i}/{len(image_files)} 枚")
        
        except Exception as e:
            print(f"❌ エラー ({image_path.name}): {e}")
            results.append({
                'filename': image_path.name,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'ぐー': 0.0, 'ちょき': 0.0, 'ぱー': 0.0}
            })
    
    return results

def generate_markdown_report(results, output_path):
    """
    推測結果をMarkdownレポートに出力
    """
    # クラス別に集計
    class_counts = {'ぐー': 0, 'ちょき': 0, 'ぱー': 0, 'ERROR': 0}
    for r in results:
        class_counts[r['predicted_class']] = class_counts.get(r['predicted_class'], 0) + 1
    
    # 信頼度別に集計
    high_conf = [r for r in results if r['confidence'] >= 0.9]
    mid_conf = [r for r in results if 0.7 <= r['confidence'] < 0.9]
    low_conf = [r for r in results if r['confidence'] < 0.7 and r['predicted_class'] != 'ERROR']
    
    # Markdown生成
    md_content = f"""# 🔍 単体テスト画像 推測結果レポート

**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

## 📊 推測サマリー

- **総画像数**: {len(results)}枚
- **処理成功**: {len([r for r in results if r['predicted_class'] != 'ERROR'])}枚
- **処理失敗**: {class_counts['ERROR']}枚

### クラス別判定結果

| クラス | 件数 | 割合 |
|--------|------|------|
| ぐー | {class_counts['ぐー']} | {class_counts['ぐー']/len(results)*100:.1f}% |
| ちょき | {class_counts['ちょき']} | {class_counts['ちょき']/len(results)*100:.1f}% |
| ぱー | {class_counts['ぱー']} | {class_counts['ぱー']/len(results)*100:.1f}% |

### 信頼度分布

| 信頼度 | 件数 | 割合 |
|--------|------|------|
| 高 (≥90%) | {len(high_conf)} | {len(high_conf)/len(results)*100:.1f}% |
| 中 (70-90%) | {len(mid_conf)} | {len(mid_conf)/len(results)*100:.1f}% |
| 低 (<70%) | {len(low_conf)} | {len(low_conf)/len(results)*100:.1f}% |

---

## 📋 全画像の判定結果

| # | 画像プレビュー | ファイル名 | 判定結果 | 信頼度 | ぐー | ちょき | ぱー |
|---|---------------|-----------|---------|--------|------|--------|------|
"""
    
    # 全結果を表に追加
    for i, r in enumerate(results, 1):
        img_path = f"{IMAGE_DIR}/{r['filename']}"
        md_content += f"| {i} | ![{r['filename']}]({img_path}) | `{r['filename']}` | **{r['predicted_class']}** | {r['confidence']*100:.2f}% | {r['probabilities']['ぐー']*100:.1f}% | {r['probabilities']['ちょき']*100:.1f}% | {r['probabilities']['ぱー']*100:.1f}% |\n"
    
    # 高信頼度の判定結果
    md_content += f"""
---

## ✅ 高信頼度判定 (≥90%)

{len(high_conf)}件の高信頼度判定がありました。

| # | 画像プレビュー | ファイル名 | 判定結果 | 信頼度 |
|---|---------------|-----------|---------|--------|
"""
    
    for i, r in enumerate(sorted(high_conf, key=lambda x: x['confidence'], reverse=True), 1):
        img_path = f"{IMAGE_DIR}/{r['filename']}"
        md_content += f"| {i} | ![{r['filename']}]({img_path}) | `{r['filename']}` | **{r['predicted_class']}** | {r['confidence']*100:.2f}% |\n"
    
    # 低信頼度の判定結果
    if low_conf:
        md_content += f"""
---

## ⚠️ 低信頼度判定 (<70%)

{len(low_conf)}件の低信頼度判定がありました（要確認）。

| # | 画像プレビュー | ファイル名 | 判定結果 | 信頼度 | ぐー | ちょき | ぱー |
|---|---------------|-----------|---------|--------|------|--------|------|
"""
        
        for i, r in enumerate(sorted(low_conf, key=lambda x: x['confidence']), 1):
            img_path = f"{IMAGE_DIR}/{r['filename']}"
            md_content += f"| {i} | ![{r['filename']}]({img_path}) | `{r['filename']}` | **{r['predicted_class']}** | {r['confidence']*100:.2f}% | {r['probabilities']['ぐー']*100:.1f}% | {r['probabilities']['ちょき']*100:.1f}% | {r['probabilities']['ぱー']*100:.1f}% |\n"
    
    # クラス別詳細
    md_content += """
---

## 📂 クラス別詳細

"""
    
    for class_name in ['ぐー', 'ちょき', 'ぱー']:
        class_results = [r for r in results if r['predicted_class'] == class_name]
        if class_results:
            avg_conf = np.mean([r['confidence'] for r in class_results])
            md_content += f"""
### {class_name} ({len(class_results)}件)

**平均信頼度**: {avg_conf*100:.2f}%

| # | 画像プレビュー | ファイル名 | 信頼度 |
|---|---------------|-----------|--------|
"""
            for i, r in enumerate(sorted(class_results, key=lambda x: x['confidence'], reverse=True), 1):
                img_path = f"{IMAGE_DIR}/{r['filename']}"
                md_content += f"| {i} | ![{r['filename']}]({img_path}) | `{r['filename']}` | {r['confidence']*100:.2f}% |\n"
    
    # フッター
    md_content += f"""
---

## 📁 ファイル情報

- **モデル**: `{MODEL_PATH}`
- **画像ディレクトリ**: `{IMAGE_DIR}`
- **総処理時間**: 自動計測未実装

---

*このレポートは `check.py` により自動生成されました*
"""
    
    # ファイルに書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✅ レポートを生成しました: {output_path}")

def main():
    """メイン処理"""
    print("=" * 60)
    print("🔍 単体テスト画像 推測システム")
    print("=" * 60)
    
    # モデルロード
    model = load_model()
    
    # 画像を推測
    results = predict_images(model, IMAGE_DIR)
    
    if not results:
        print("\n❌ 処理する画像がありませんでした")
        return
    
    # Markdownレポート生成
    generate_markdown_report(results, OUTPUT_MD)
    
    # サマリー表示
    print("\n" + "=" * 60)
    print("📊 処理完了サマリー")
    print("=" * 60)
    print(f"総画像数: {len(results)}枚")
    
    class_counts = {}
    for r in results:
        class_counts[r['predicted_class']] = class_counts.get(r['predicted_class'], 0) + 1
    
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}枚 ({count/len(results)*100:.1f}%)")
    
    print(f"\n📄 詳細レポート: {OUTPUT_MD}")
    print("=" * 60)

if __name__ == "__main__":
    main()
