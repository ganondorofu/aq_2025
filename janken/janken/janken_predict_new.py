import os
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
    if os.path.exists("model.keras"):
        model = tf.keras.models.load_model("model.keras")
        print("model.kerasを読み込みました")
    elif os.path.exists("model.h5"):
        model = tf.keras.models.load_model("model.h5")
        print("model.h5を読み込みました")
    else:
        raise FileNotFoundError("モデルファイルが見つかりません")

    # 予測実施
    pred_confidence = model.predict(test_ds_processed)
    pred_class = np.argmax(pred_confidence, axis=1)

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


if __name__ == "__main__":
    main()