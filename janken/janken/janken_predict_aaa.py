"""
じゃんけん判定AI - 予測・評価・画像プレビュー付きレポート生成
画像のプレビューが見れるMarkdownレポートを生成します
"""

import os
import shutil
import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from janken_train_new import target_size
from janken_train_new import batch_size
from janken_train_new import preprocessing_function


def main():
    # 評価用データセット作成
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "img_test",
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # クラス名を取得
    class_names = test_ds.class_names
    print(f"クラス名: {class_names}")
    
    # クラス名のマッピング（日本語）
    class_map = {}
    for i, name in enumerate(class_names):
        if 'gu' in name.lower():
            class_map[i] = 'ぐー'
        elif 'tyoki' in name.lower() or 'choki' in name.lower():
            class_map[i] = 'ちょき'
        elif 'pa' in name.lower():
            class_map[i] = 'ぱー'
        else:
            class_map[i] = name
    
    # プリプロセッシングを適用
    test_ds_processed = test_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds_processed = test_ds_processed.prefetch(tf.data.AUTOTUNE)

    # 正解ラベルを取得
    true_labels = []
    for _, labels in test_ds:
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
    true_labels = np.array(true_labels)
    
    # 画像パスを取得
    image_paths = test_ds.file_paths

    # 学習済みモデルロード（.kerasを優先、なければ.h5）
    model_path = "model_with_subdirs.keras"
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"{model_path}を読み込みました")
    elif os.path.exists("model.h5"):
        model = tf.keras.models.load_model("model.h5")
        print("model.h5を読み込みました")
    else:
        raise FileNotFoundError("モデルファイルが見つかりません")

    # 予測実施
    pred_confidence = model.predict(test_ds_processed)
    pred_class = np.argmax(pred_confidence, axis=1)
    confidences = np.max(pred_confidence, axis=1)

    # 予測結果ファイル出力
    print("\n予測結果:")
    print(pred_class)
    np.savetxt("result.csv", pred_class, fmt="%d")

    # ========== 評価指標の計算 ==========
    print("\n" + "="*60)
    print("評価結果")
    print("="*60)
    
    # 正解率
    accuracy = accuracy_score(true_labels, pred_class)
    print(f"\n正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 適合率、再現率、F値（マクロ平均とクラスごと）
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_class, average=None, labels=[0, 1, 2]
    )
    
    print("\nクラスごとの評価指標:")
    print("-" * 60)
    print(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12}")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}")
    
    # マクロ平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_class, average='macro'
    )
    print("-" * 60)
    print(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    
    # 重み付き平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_class, average='weighted'
    )
    print(f"{'重み付き平均':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")
    
    # 混同行列
    cm = confusion_matrix(true_labels, pred_class, labels=[0, 1, 2])
    print("\n混同行列:")
    print("-" * 60)
    print(f"{'':>15}", end="")
    for class_name in class_names:
        print(f"{class_name:>15}", end="")
    print()
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>15}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15}", end="")
        print()
    
    # scikit-learnの詳細レポート
    print("\n詳細な分類レポート:")
    print("-" * 60)
    print(classification_report(true_labels, pred_class, target_names=class_names, digits=4))
    
    # 評価結果をファイルに保存
    with open("evaluation_result.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("評価結果\n")
        f.write("="*60 + "\n\n")
        f.write(f"正解率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("クラスごとの評価指標:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'クラス':<15} {'適合率':<12} {'再現率':<12} {'F値':<12} {'サンプル数':<12}\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<12}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'マクロ平均':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}\n")
        f.write(f"{'重み付き平均':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}\n\n")
        f.write("混同行列:\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(true_labels, pred_class, target_names=class_names, digits=4))
    
    print("\n評価結果を evaluation_result.txt に保存しました")
    print("="*60)
    
    # ========== 画像プレビュー付きレポート生成 ==========
    print("\n画像プレビュー付きレポートを生成中...")
    
    # レポート用ディレクトリ作成
    report_dir = "prediction_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 画像を分類してコピー
    failed_dir = os.path.join(report_dir, "failed_images")
    correct_dir = os.path.join(report_dir, "correct_images")
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    # 失敗ケースと成功ケースの収集
    failed_predictions = []
    correct_predictions = []
    
    for i, (true_cls, pred_cls, conf, img_path) in enumerate(zip(true_labels, pred_class, confidences, image_paths)):
        img_info = {
            'index': i + 1,
            'filename': Path(img_path).name,
            'original_path': img_path,
            'true_class': class_map[true_cls],
            'pred_class': class_map[pred_cls],
            'true_class_id': true_cls,
            'pred_class_id': pred_cls,
            'confidence': conf,
            'is_correct': true_cls == pred_cls
        }
        
        if true_cls == pred_cls:
            correct_predictions.append(img_info)
        else:
            failed_predictions.append(img_info)
    
    # 失敗画像をコピー
    for failure in failed_predictions:
        src_path = failure['original_path']
        true_class = failure['true_class']
        pred_class = failure['pred_class']
        
        pattern_dir = os.path.join(failed_dir, f"実際_{true_class}_予測_{pred_class}")
        os.makedirs(pattern_dir, exist_ok=True)
        
        filename = failure['filename']
        base_name = Path(filename).stem
        ext = Path(filename).suffix
        new_filename = f"{base_name}_conf{failure['confidence']:.3f}{ext}"
        
        dst_path = os.path.join(pattern_dir, new_filename)
        failure['report_path'] = os.path.relpath(dst_path, report_dir)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # 正解画像もクラスごとにコピー
    for correct in correct_predictions:
        src_path = correct['original_path']
        class_name = correct['true_class']
        
        class_dir = os.path.join(correct_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        filename = correct['filename']
        base_name = Path(filename).stem
        ext = Path(filename).suffix
        new_filename = f"{base_name}_conf{correct['confidence']:.3f}{ext}"
        
        dst_path = os.path.join(class_dir, new_filename)
        correct['report_path'] = os.path.relpath(dst_path, report_dir)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Markdownレポート生成
    md_path = os.path.join(report_dir, "PREDICTION_REPORT.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 🔍 じゃんけん判定AI - 予測結果レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # サマリー
        f.write("## 📊 評価サマリー\n\n")
        f.write(f"- **総合精度**: {accuracy*100:.2f}%\n")
        f.write(f"- **テスト総数**: {len(true_labels)}枚\n")
        f.write(f"- **正解**: {len(correct_predictions)}枚 ✅\n")
        f.write(f"- **不正解**: {len(failed_predictions)}枚 ❌\n")
        f.write(f"- **成功率**: {len(correct_predictions)/len(true_labels)*100:.2f}%\n\n")
        
        # クラス別精度
        f.write("### クラス別精度\n\n")
        f.write("| クラス | Precision | Recall | F1-Score | サンプル数 |\n")
        f.write("|--------|-----------|--------|----------|------------|\n")
        for i, class_name in enumerate(class_names):
            display_name = class_map[i]
            f.write(f"| {display_name} | {precision[i]*100:.2f}% | {recall[i]*100:.2f}% | "
                   f"{f1[i]*100:.2f}% | {int(support[i])} |\n")
        f.write("\n---\n\n")
        
        # 混同行列
        f.write("## 📈 混同行列\n\n")
        f.write("| 実際＼予測 |")
        for i in range(len(class_names)):
            f.write(f" {class_map[i]} |")
        f.write("\n|")
        f.write("------------|" * (len(class_names) + 1))
        f.write("\n")
        
        for i in range(len(class_names)):
            f.write(f"| **{class_map[i]}** |")
            for j in range(len(class_names)):
                count = cm[i, j]
                percent = (count / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
                f.write(f" {count} ({percent:.1f}%) |")
            f.write("\n")
        
        f.write("\n---\n\n")
        
        # 失敗ケース（画像プレビュー付き）
        if failed_predictions:
            f.write(f"## ❌ 失敗ケース詳細 ({len(failed_predictions)}件)\n\n")
            
            # 誤分類パターンごとに整理
            failure_patterns = {}
            for failure in failed_predictions:
                pattern = f"{failure['true_class']} → {failure['pred_class']}"
                if pattern not in failure_patterns:
                    failure_patterns[pattern] = []
                failure_patterns[pattern].append(failure)
            
            # パターンサマリー
            f.write("### 誤分類パターンサマリー\n\n")
            f.write("| 実際のクラス | 予測クラス | 件数 | 割合 |\n")
            f.write("|--------------|------------|------|------|\n")
            for pattern, pattern_failures in sorted(failure_patterns.items(), key=lambda x: len(x[1]), reverse=True):
                true_cls, pred_cls = pattern.split(' → ')
                count = len(pattern_failures)
                percent = count / len(failed_predictions) * 100
                f.write(f"| {true_cls} | {pred_cls} | {count} | {percent:.1f}% |\n")
            f.write("\n")
            
            # パターンごとに画像プレビュー
            for pattern, pattern_failures in sorted(failure_patterns.items()):
                true_cls, pred_cls = pattern.split(' → ')
                f.write(f"### {pattern} ({len(pattern_failures)}件)\n\n")
                
                # 信頼度の高い順にソート
                sorted_failures = sorted(pattern_failures, key=lambda x: x['confidence'], reverse=True)
                
                # テーブルで一覧
                f.write("| # | 画像 | ファイル名 | 信頼度 |\n")
                f.write("|---|------|-----------|--------|\n")
                
                for i, failure in enumerate(sorted_failures, 1):
                    f.write(f"| {i} | ![]({failure['report_path']}) | `{failure['filename']}` | {failure['confidence']*100:.2f}% |\n")
                
                f.write("\n")
            
            f.write("---\n\n")
            
            # 高信頼度での誤検出
            f.write("## ⚠️ 高信頼度での誤検出 TOP 10\n\n")
            f.write("モデルが確信を持って間違えたケース（要注意）:\n\n")
            
            sorted_all_failures = sorted(failed_predictions, key=lambda x: x['confidence'], reverse=True)[:10]
            
            f.write("| 順位 | 画像プレビュー | ファイル名 | 実際 | 予測 | 信頼度 |\n")
            f.write("|------|---------------|-----------|------|------|--------|\n")
            
            for i, failure in enumerate(sorted_all_failures, 1):
                f.write(f"| {i} | ![]({failure['report_path']}) | `{failure['filename']}` | "
                       f"{failure['true_class']} | {failure['pred_class']} | {failure['confidence']*100:.2f}% |\n")
            
            f.write("\n")
        else:
            f.write("## 🎉 失敗ケースなし\n\n")
            f.write("全てのテストケースで正しく予測されました！\n\n")
        
        f.write("---\n\n")
        
        # 正解ケース（画像プレビュー付き）
        if correct_predictions:
            f.write(f"## ✅ 正解ケース ({len(correct_predictions)}件)\n\n")
            
            # クラスごとに整理
            correct_by_class = {}
            for correct in correct_predictions:
                cls = correct['true_class']
                if cls not in correct_by_class:
                    correct_by_class[cls] = []
                correct_by_class[cls].append(correct)
            
            for cls, cls_correct in sorted(correct_by_class.items()):
                f.write(f"### {cls} ({len(cls_correct)}件)\n\n")
                
                # 信頼度の高い順に表示（最大10件）
                sorted_correct = sorted(cls_correct, key=lambda x: x['confidence'], reverse=True)[:10]
                
                f.write("| # | 画像プレビュー | ファイル名 | 信頼度 |\n")
                f.write("|---|---------------|-----------|--------|\n")
                
                for i, correct in enumerate(sorted_correct, 1):
                    f.write(f"| {i} | ![]({correct['report_path']}) | `{correct['filename']}` | {correct['confidence']*100:.2f}% |\n")
                
                if len(cls_correct) > 10:
                    f.write(f"\n*他 {len(cls_correct) - 10}件は省略*\n")
                
                f.write("\n")
        
        f.write("---\n\n")
        
        # 改善提案
        f.write("## 💡 改善提案\n\n")
        
        if failed_predictions:
            worst_class_idx = np.argmin(f1)
            worst_class = class_map[worst_class_idx]
            worst_f1 = f1[worst_class_idx] * 100
            
            f.write(f"1. **{worst_class}クラスの精度改善** (F1スコア: {worst_f1:.2f}%)\n")
            f.write(f"   - このクラスのトレーニングデータを増やす\n")
            f.write(f"   - データ拡張を強化する\n\n")
            
            if failure_patterns:
                top_pattern = max(failure_patterns.items(), key=lambda x: len(x[1]))
                f.write(f"2. **{top_pattern[0]}の誤分類対策** ({len(top_pattern[1])}件)\n")
                f.write(f"   - この誤分類パターンが最も多い\n")
                f.write(f"   - 上記の画像を確認して、共通の特徴を分析\n\n")
            
            f.write("3. **高信頼度での誤検出画像を確認**\n")
            f.write("   - モデルが確信を持って間違えている画像を重点的に分析\n")
            f.write("   - ラベルが正しいか再確認\n\n")
        
        f.write("---\n\n")
        
        # ファイル構成
        f.write("## 📁 ファイル構成\n\n")
        f.write("```\n")
        f.write("prediction_report/\n")
        f.write("├── PREDICTION_REPORT.md (このファイル)\n")
        f.write("├── failed_images/\n")
        for pattern in sorted(failure_patterns.keys()):
            true_cls, pred_cls = pattern.split(' → ')
            folder = f"実際_{true_cls}_予測_{pred_cls}"
            count = len(failure_patterns[pattern])
            f.write(f"│   ├── {folder}/ ({count}枚)\n")
        f.write("└── correct_images/\n")
        for cls in sorted(correct_by_class.keys()):
            count = len(correct_by_class[cls])
            f.write(f"    ├── {cls}/ ({count}枚)\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        f.write("*このレポートは `janken_predict_aaa.py` により自動生成されました*\n")
    
    print(f"✓ 画像プレビュー付きレポート生成完了: {md_path}")
    print(f"✓ レポートフォルダ: {report_dir}")
    print(f"  - 失敗画像: {len(failed_predictions)}枚")
    print(f"  - 正解画像: {len(correct_predictions)}枚")
    print("="*60)
    
    # テキスト形式の詳細リストも生成
    failed_list_path = 'failed_images_list.txt'
    with open(failed_list_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("間違えた画像一覧\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"総テスト数: {len(true_labels)}枚\n")
        f.write(f"正解数: {len(correct_predictions)}枚 ({len(correct_predictions)/len(true_labels)*100:.2f}%)\n")
        f.write(f"不正解数: {len(failed_predictions)}枚 ({len(failed_predictions)/len(true_labels)*100:.2f}%)\n")
        f.write("=" * 80 + "\n\n")
        
        if failed_predictions:
            failure_patterns = {}
            for failure in failed_predictions:
                pattern = f"{failure['true_class']} → {failure['pred_class']}"
                if pattern not in failure_patterns:
                    failure_patterns[pattern] = []
                failure_patterns[pattern].append(failure)
            
            for pattern, pattern_failures in sorted(failure_patterns.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"\n【{pattern}】 {len(pattern_failures)}件\n")
                f.write("-" * 80 + "\n")
                
                sorted_failures = sorted(pattern_failures, key=lambda x: x['confidence'], reverse=True)
                
                for failure in sorted_failures:
                    f.write(f"  {failure['index']:3d}. {failure['filename']:<40} ")
                    f.write(f"実際:{failure['true_class']:>6} → 予測:{failure['pred_class']:>6} ")
                    f.write(f"信頼度:{failure['confidence']*100:6.2f}%\n")
                    f.write(f"       パス: {failure['original_path']}\n")
        else:
            f.write("\n🎉 間違えた画像はありません！全て正解です！\n")
    
    print(f"✓ テキスト形式の失敗リスト: {failed_list_path}")
    print("="*60)


if __name__ == "__main__":
    main()
