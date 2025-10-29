"""
ハードネガティブマイニング版の学習スクリプト（動的版）
誤分類されやすい画像（ハードネガティブ）を学習中に特定して重点的に学習

改善点:
1. 学習途中で検証データを使って難しい画像を特定
2. 次のエポックでそれらの画像を重点的に学習
3. クラスの重み付けで「ちょき」クラスを強化
4. 2段階学習: Stage 1で失敗画像を特定 → Stage 2で集中学習
"""

import os
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from PIL import Image


# ハイパーパラメーター設定
target_size = 224
batch_size = 16
epochs_stage1 = 30  # Stage 1: 失敗画像を特定
epochs_stage2 = 50  # Stage 2: 失敗画像を重点学習
learning_rate = 0.0001

# ハードネガティブマイニング設定
HARD_NEGATIVE_WEIGHT = 3.0  # 難しい画像のサンプリング重み（3倍）
CONFIDENCE_THRESHOLD = 0.7  # この信頼度以下を「難しい画像」と判定
CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 1.0}  # クラス重み (ちょきを1.5倍)

MAX_SUBDIR_IMAGES = None
preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# ============================================================
# カスタムブレ対策レイヤー
# ============================================================

class RandomDefocus(tf.keras.layers.Layer):
    """ピンぼけシミュレーション"""
    def __init__(self, max_strength=0.3, **kwargs):
        super().__init__(**kwargs)
        self.max_strength = max_strength
    
    def call(self, images, training=None):
        if not training:
            return images
        
        if tf.random.uniform(()) > 0.3:
            return images
        
        strength = tf.random.uniform((), 0, self.max_strength)
        noise = tf.random.normal(tf.shape(images), mean=0, stddev=strength * 10)
        return tf.clip_by_value(images + noise, 0, 255)


# ============================================================
# ハードネガティブマイニング用の関数（動的版）
# ============================================================

def identify_hard_negatives(model, dataset, dataset_dir):
    """
    学習済みモデルを使って難しい画像を特定
    
    Returns:
        hard_negative_files: 難しい画像のファイル名セット
    """
    print("\n🔍 難しい画像を特定中...")
    
    hard_negative_files = set()
    class_names = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))])
    
    # 各クラスのディレクトリから画像を評価
    total_images = 0
    hard_images = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        
        for img_file in image_files:
            # "hard_" プレフィックスは除外（既に複製されたもの）
            if img_file.startswith('hard_'):
                continue
            
            img_path = os.path.join(class_path, img_file)
            
            try:
                # 画像を読み込んで前処理
                img = Image.open(img_path).convert('RGB')
                img = img.resize((target_size, target_size))
                img_array = np.array(img)
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # 予測
                predictions = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                total_images += 1
                
                # 条件1: 誤分類
                # 条件2: 信頼度が低い（曖昧な画像）
                is_misclassified = (predicted_class != class_idx)
                is_low_confidence = (confidence < CONFIDENCE_THRESHOLD)
                
                if is_misclassified or is_low_confidence:
                    # オリジナルのファイル名を抽出（サブディレクトリプレフィックスを除去）
                    original_name = img_file
                    if '_' in img_file:
                        # "subdir_filename.jpg" → "filename.jpg" を試す
                        parts = img_file.split('_', 1)
                        if len(parts) > 1:
                            original_name = parts[1]
                    
                    hard_negative_files.add(img_file)
                    hard_images += 1
                    
                    if is_misclassified:
                        print(f"  ❌ 誤分類: {class_name}/{img_file} → 予測: {class_names[predicted_class]} (信頼度: {confidence:.2%})")
                    else:
                        print(f"  ⚠️ 低信頼度: {class_name}/{img_file} (信頼度: {confidence:.2%})")
            
            except Exception as e:
                print(f"  ⚠️ エラー ({img_file}): {e}")
                continue
    
    print(f"\n� 特定結果:")
    print(f"  総画像数: {total_images}枚")
    print(f"  難しい画像: {hard_images}枚 ({hard_images/total_images*100:.1f}%)")
    print(f"  識別されたファイル: {len(hard_negative_files)}個")
    
    # 結果を保存
    with open('hard_negatives_identified.txt', 'w', encoding='utf-8') as f:
        for filename in sorted(hard_negative_files):
            f.write(f"{filename}\n")
    print(f"  ✅ 保存完了: hard_negatives_identified.txt")
    
    return hard_negative_files


def create_weighted_dataset(base_dir, failed_images, max_subdir_images=None):
    """
    ハードネガティブ画像の重みを増やしたデータセットを作成
    """
    merged_dir = f"{base_dir}_merged_weighted"
    
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    
    os.makedirs(merged_dir, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    hard_negative_count = 0
    normal_count = 0
    
    for class_dir in class_dirs:
        source_class_path = os.path.join(base_dir, class_dir)
        target_class_path = os.path.join(merged_dir, class_dir)
        os.makedirs(target_class_path, exist_ok=True)
        
        # クラスディレクトリ直下の画像
        for file in os.listdir(source_class_path):
            file_path = os.path.join(source_class_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    # 通常コピー
                    shutil.copy2(file_path, os.path.join(target_class_path, file))
                    normal_count += 1
                    
                    # ハードネガティブの場合は複数回コピー（重み付け）
                    if file in failed_images:
                        for i in range(int(HARD_NEGATIVE_WEIGHT - 1)):
                            duplicate_name = f"hard_{i}_{file}"
                            shutil.copy2(file_path, os.path.join(target_class_path, duplicate_name))
                            hard_negative_count += 1
        
        # サブディレクトリ内の画像
        for subdir in os.listdir(source_class_path):
            subdir_path = os.path.join(source_class_path, subdir)
            if os.path.isdir(subdir_path):
                subdir_images = []
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                            subdir_images.append((file_path, file))
                
                if max_subdir_images is None or len(subdir_images) <= max_subdir_images:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚すべて使用")
                else:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚 → {max_subdir_images}枚をランダム選択")
                    subdir_images = random.sample(subdir_images, max_subdir_images)
                
                for file_path, file in subdir_images:
                    new_filename = f"{subdir}_{file}"
                    shutil.copy2(file_path, os.path.join(target_class_path, new_filename))
                    normal_count += 1
                    
                    # ハードネガティブの場合は複数回コピー
                    if file in failed_images:
                        for i in range(int(HARD_NEGATIVE_WEIGHT - 1)):
                            duplicate_name = f"hard_{i}_{subdir}_{file}"
                            shutil.copy2(file_path, os.path.join(target_class_path, duplicate_name))
                            hard_negative_count += 1
    
    print(f"\n🎯 ハードネガティブマイニング統計:")
    print(f"  通常画像: {normal_count}枚")
    print(f"  ハードネガティブ複製: {hard_negative_count}枚 (重み: {HARD_NEGATIVE_WEIGHT}x)")
    print(f"  合計: {normal_count + hard_negative_count}枚")
    
    return merged_dir


def create_merged_dataset(base_dir, max_subdir_images=MAX_SUBDIR_IMAGES):
    """通常のマージ（テストデータ用）"""
    merged_dir = f"{base_dir}_merged"
    
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    
    os.makedirs(merged_dir, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_dir in class_dirs:
        source_class_path = os.path.join(base_dir, class_dir)
        target_class_path = os.path.join(merged_dir, class_dir)
        os.makedirs(target_class_path, exist_ok=True)
        
        for file in os.listdir(source_class_path):
            file_path = os.path.join(source_class_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    shutil.copy2(file_path, os.path.join(target_class_path, file))
        
        for subdir in os.listdir(source_class_path):
            subdir_path = os.path.join(source_class_path, subdir)
            if os.path.isdir(subdir_path):
                subdir_images = []
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                            subdir_images.append((file_path, file))
                
                if max_subdir_images is None or len(subdir_images) <= max_subdir_images:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚すべて使用")
                else:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚 → {max_subdir_images}枚をランダム選択")
                    subdir_images = random.sample(subdir_images, max_subdir_images)
                
                for file_path, file in subdir_images:
                    new_filename = f"{subdir}_{file}"
                    shutil.copy2(file_path, os.path.join(target_class_path, new_filename))
    
    return merged_dir


def count_images(directory):
    """ディレクトリ内の画像数をカウント"""
    print(f"\n📊 {directory} の画像数:")
    print("=" * 60)
    total = 0
    class_dirs = sorted([d for d in os.listdir(directory) 
                        if os.path.isdir(os.path.join(directory, d))])
    
    for class_dir in class_dirs:
        class_path = os.path.join(directory, class_dir)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        count = len(image_files)
        total += count
        print(f"  {class_dir}: {count}枚")
    
    print(f"  {'合計'}: {total}枚")
    print("=" * 60)
    return total


def plot_result(history):
    """学習結果のグラフを描画"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train_accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="val_accuracy", marker="s")
    plt.title("Accuracy (Hard Negative Mining)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("mlp_graph_accuracy_hard_negative.png")
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train_loss", marker="o")
    plt.plot(history.history["val_loss"], label="val_loss", marker="s")
    plt.title("Loss (Hard Negative Mining)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.savefig("mlp_graph_loss_hard_negative.png")


def _main():
    print("=" * 60)
    print("🎯 ハードネガティブマイニング学習（動的2段階版）")
    print("=" * 60)
    print("\n特徴:")
    print("  📍 Stage 1: 通常学習で難しい画像を特定")
    print("  🎯 Stage 2: 難しい画像を重点的に再学習")
    print(f"  ✅ ハードネガティブの重み: {HARD_NEGATIVE_WEIGHT}x")
    print(f"  ✅ クラス重み: ぐー={CLASS_WEIGHTS[0]}, ちょき={CLASS_WEIGHTS[1]}, ぱー={CLASS_WEIGHTS[2]}")
    print(f"  ✅ 低信頼度閾値: {CONFIDENCE_THRESHOLD}")
    print("=" * 60)
    
    # スクリプトの場所を基準にパスを設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "img_train")
    test_dir = os.path.join(script_dir, "img_test")
    
    # ========================================
    # Stage 1: 通常学習で難しい画像を特定
    # ========================================
    print("\n" + "=" * 60)
    print("🚀 Stage 1: 通常学習開始")
    print("=" * 60)
    
    print("\n🔄 データセットを準備中...")
    merged_train_dir = create_merged_dataset(train_dir)
    merged_test_dir = create_merged_dataset(test_dir)
    
    count_images(merged_train_dir)
    count_images(merged_test_dir)
    
    print("\n📚 データセットを読み込み中...")
    
    # 学習用データセット作成
    train_ds = tf.keras.utils.image_dataset_from_directory(
        merged_train_dir,
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    # 評価用データセット作成
    test_ds = tf.keras.utils.image_dataset_from_directory(
        merged_test_dir,
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # データ拡張（GA最適化済みパラメータ）
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.117),
        tf.keras.layers.RandomZoom(0.021),
        tf.keras.layers.RandomTranslation(0.094, 0.094),
        tf.keras.layers.RandomBrightness(0.447),
        tf.keras.layers.RandomContrast(0.428),
        tf.keras.layers.GaussianNoise(0.130),
        RandomDefocus(max_strength=0.02),
    ], name='optimized_augmentation')
    
    # 前処理を適用
    train_ds_stage1 = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds_stage1 = train_ds_stage1.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds_stage1 = train_ds_stage1.prefetch(tf.data.AUTOTUNE)
    
    test_ds_processed = test_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds_processed = test_ds_processed.prefetch(tf.data.AUTOTUNE)

    print("\n🏗️ Stage 1モデルを構築中...")
    
    # モデル作成
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(target_size, target_size, 3),
        include_top=False,
        weights="imagenet"
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model_stage1 = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    model_stage1.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
    ]

    print("\n🚀 Stage 1 学習開始...")
    print(f"  - 最大エポック数: {epochs_stage1}")
    print(f"  - 目的: 難しい画像を特定")
    print("=" * 60)
    
    history_stage1 = model_stage1.fit(
        train_ds_stage1,
        validation_data=test_ds_processed,
        epochs=epochs_stage1,
        callbacks=callbacks_stage1,
        class_weight=CLASS_WEIGHTS,
        verbose=2
    )
    
    print("\n✅ Stage 1 完了!")
    stage1_val_acc = max(history_stage1.history['val_accuracy'])
    print(f"📈 Stage 1 最高検証精度: {stage1_val_acc*100:.2f}%")
    
    # 難しい画像を特定
    hard_negative_files = identify_hard_negatives(model_stage1, train_ds, merged_train_dir)
    
    # Stage 1モデルを保存
    model_stage1.save("model_stage1.keras", include_optimizer=False)
    print("💾 Stage 1モデル保存: model_stage1.keras")
    
    # ========================================
    # Stage 2: ハードネガティブを重点学習
    # ========================================
    print("\n" + "=" * 60)
    print("🎯 Stage 2: ハードネガティブ重点学習開始")
    print("=" * 60)
    
    # ハードネガティブを重み付けした学習データセット作成
    print("\n🔄 ハードネガティブ重み付けデータセットを準備中...")
    
    # 一度クリーンアップ
    if os.path.exists(merged_train_dir):
        shutil.rmtree(merged_train_dir)
    
    # 重み付けデータセット作成
    merged_train_dir_weighted = create_weighted_dataset(train_dir, hard_negative_files)
    
    count_images(merged_train_dir_weighted)
    
    # 新しいデータセット作成
    train_ds_stage2 = tf.keras.utils.image_dataset_from_directory(
        merged_train_dir_weighted,
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    # 前処理を適用
    train_ds_stage2 = train_ds_stage2.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds_stage2 = train_ds_stage2.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds_stage2 = train_ds_stage2.prefetch(tf.data.AUTOTUNE)
    
    print("\n🏗️ Stage 2モデルを構築中（Stage 1から継続）...")
    
    # Stage 1のモデルをベースに継続学習
    model_stage2 = model_stage1  # 同じモデルを使用
    
    # 学習率を少し下げる
    model_stage2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate * 0.5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_hard_negative.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]

    print("\n🚀 Stage 2 学習開始...")
    print(f"  - 最大エポック数: {epochs_stage2}")
    print(f"  - ハードネガティブ: {len(hard_negative_files)}個")
    print(f"  - 学習率: {learning_rate * 0.5}")
    print("=" * 60)
    
    history_stage2 = model_stage2.fit(
        train_ds_stage2,
        validation_data=test_ds_processed,
        epochs=epochs_stage2,
        callbacks=callbacks_stage2,
        class_weight=CLASS_WEIGHTS,
        verbose=2
    )

    print("\n💾 最終モデルを保存中...")
    
    model_stage2.save("model_hard_negative.keras", include_optimizer=False)
    print("✅ モデル保存完了: model_hard_negative.keras")

    print("\n📊 学習結果のグラフを生成中...")
    
    # 2段階の学習履歴を統合してグラフ化
    combined_history = {
        'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
        'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
        'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
        'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss'],
    }
    
    # グラフを描画（Stage境界を表示）
    plt.figure(figsize=(15, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(combined_history['accuracy']) + 1)
    plt.plot(epochs_range, combined_history["accuracy"], label="train_accuracy", marker="o")
    plt.plot(epochs_range, combined_history["val_accuracy"], label="val_accuracy", marker="s")
    plt.axvline(x=len(history_stage1.history['accuracy']), color='r', linestyle='--', label='Stage 1→2')
    plt.title("Accuracy (2-Stage Hard Negative Mining)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, combined_history["loss"], label="train_loss", marker="o")
    plt.plot(epochs_range, combined_history["val_loss"], label="val_loss", marker="s")
    plt.axvline(x=len(history_stage1.history['loss']), color='r', linestyle='--', label='Stage 1→2')
    plt.title("Loss (2-Stage Hard Negative Mining)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("mlp_graph_hard_negative_2stage.png")
    print("✅ グラフ保存完了: mlp_graph_hard_negative_2stage.png")
    
    # 学習結果をJSONで保存
    stage2_val_acc = max(history_stage2.history['val_accuracy'])
    result_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hard_negative_weight': HARD_NEGATIVE_WEIGHT,
        'class_weights': CLASS_WEIGHTS,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'hard_negatives_count': len(hard_negative_files),
        'stage1': {
            'epochs': len(history_stage1.history['val_accuracy']),
            'best_val_accuracy': float(max(history_stage1.history['val_accuracy'])),
            'final_val_accuracy': float(history_stage1.history['val_accuracy'][-1]),
        },
        'stage2': {
            'epochs': len(history_stage2.history['val_accuracy']),
            'best_val_accuracy': float(max(history_stage2.history['val_accuracy'])),
            'final_val_accuracy': float(history_stage2.history['val_accuracy'][-1]),
        },
        'improvement': float(stage2_val_acc - stage1_val_acc)
    }
    
    with open('hard_negative_training_result.json', 'w') as f:
        json.dump(result_summary, f, indent=2)
    print("✅ 学習結果を保存: hard_negative_training_result.json")
    
    # 一時ディレクトリをクリーンアップ
    print("\n🧹 一時ファイルをクリーンアップ中...")
    if os.path.exists(merged_train_dir_weighted):
        shutil.rmtree(merged_train_dir_weighted)
    if os.path.exists(merged_test_dir):
        shutil.rmtree(merged_test_dir)
    print("✅ クリーンアップ完了")
    
    print("\n" + "=" * 60)
    print("🎉 2段階ハードネガティブマイニング学習が完了しました!")
    print("=" * 60)
    print(f"\n📈 Stage 1 最高精度: {stage1_val_acc*100:.2f}%")
    print(f"📈 Stage 2 最高精度: {stage2_val_acc*100:.2f}%")
    print(f"🚀 改善幅: +{result_summary['improvement']*100:.2f}%")
    print(f"🎯 難しい画像: {len(hard_negative_files)}個を特定・強化学習")


if __name__ == "__main__":
    _main()
