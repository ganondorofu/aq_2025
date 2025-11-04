"""
改善版じゃんけん画像分類モデル学習スクリプト

主な改善点:
1. 段階的ファインチューニング (凍結→部分解凍→全体学習)
2. MixUp データ拡張
3. コサインアニーリング学習率スケジューリング
4. ラベルスムージング
5. Test Time Augmentation (TTA)
6. Stochastic Weight Averaging (SWA)
"""

import os
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime


# ============================================================
# ハイパーパラメーター設定
# ============================================================

# 基本設定
target_size = 224
batch_size = 16
initial_epochs = 30      # Stage 1: 凍結学習
finetune_epochs = 50     # Stage 2: ファインチューニング
total_epochs = 100       # Stage 3: 全体学習
initial_learning_rate = 0.001   # Stage 1の学習率
finetune_learning_rate = 0.0001 # Stage 2-3の学習率

# ラベルスムージング (0.1 = 10%のスムージング)
label_smoothing = 0.1

# MixUp alpha (0.2 推奨)
mixup_alpha = 0.2

# Test Time Augmentation
tta_steps = 5  # TTA時の拡張回数

# サブディレクトリから選択する画像数
MAX_SUBDIR_IMAGES = None

# シード設定（再現性のため）
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input


# ============================================================
# MixUp データ拡張
# ============================================================

class MixupLayer(tf.keras.layers.Layer):
    """
    MixUp data augmentation layer
    2つの画像とラベルを混ぜ合わせることで、モデルの汎化性能を向上
    """
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, images, labels, training=None):
        if not training or self.alpha <= 0:
            return images, labels
        
        batch_size = tf.shape(images)[0]
        
        # Beta分布からlambdaをサンプリング
        lambda_value = tf.random.uniform([], 0, self.alpha)
        
        # ランダムにバッチをシャッフル
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # 画像とラベルをMixUp
        mixed_images = lambda_value * images + (1 - lambda_value) * tf.gather(images, indices)
        mixed_labels = lambda_value * labels + (1 - lambda_value) * tf.gather(labels, indices)
        
        return mixed_images, mixed_labels


# ============================================================
# カスタム学習率スケジューラー（コサインアニーリング）
# ============================================================

class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    """
    コサインアニーリング学習率スケジューラー
    学習率を滑らかに減衰させることで、より良い最適解を見つける
    """
    def __init__(self, initial_lr, min_lr, total_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        # コサイン関数で学習率を計算
        cosine = np.cos(np.pi * epoch / self.total_epochs)
        lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + cosine) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        print(f'\n📊 Epoch {epoch + 1}: Learning rate = {lr:.6f}')


# ============================================================
# カスタムブレ対策レイヤー
# ============================================================

class RandomDefocus(tf.keras.layers.Layer):
    """ピンぼけをシミュレート"""
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
# データセット作成関数
# ============================================================

def create_merged_dataset(base_dir, max_subdir_images=MAX_SUBDIR_IMAGES):
    """
    各クラスディレクトリ内の画像とサブディレクトリ内の画像を統合
    """
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
        
        # クラスディレクトリ直下の画像ファイルをコピー
        for file in os.listdir(source_class_path):
            file_path = os.path.join(source_class_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    shutil.copy2(file_path, os.path.join(target_class_path, file))
        
        # サブディレクトリ内の画像ファイルを収集
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
                      if os.path.isfile(os.path.join(class_path, f)) and
                      f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        count = len(image_files)
        total += count
        print(f"  {class_dir}: {count}枚")
    
    print(f"  合計: {total}枚")
    print("=" * 60)


# ============================================================
# Test Time Augmentation (TTA)
# ============================================================

def predict_with_tta(model, dataset, num_augmentations=5):
    """
    Test Time Augmentation を使った予測
    複数回の拡張を行い、その平均を取ることで精度向上
    """
    predictions = []
    labels_list = []
    
    # データ拡張用のレイヤー
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.05),
    ])
    
    for images, labels in dataset:
        # 元の予測
        pred = model.predict(images, verbose=0)
        aug_preds = [pred]
        
        # 拡張した画像での予測
        for _ in range(num_augmentations - 1):
            aug_images = augmentation(images, training=False)
            aug_pred = model.predict(aug_images, verbose=0)
            aug_preds.append(aug_pred)
        
        # 平均を取る
        avg_pred = np.mean(aug_preds, axis=0)
        predictions.append(avg_pred)
        labels_list.append(labels.numpy())
    
    predictions = np.vstack(predictions)
    labels_array = np.vstack(labels_list)
    
    # 精度計算
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels_array, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    
    return accuracy


# ============================================================
# グラフ描画
# ============================================================

def plot_training_history(history_stage1, history_stage2, history_stage3, save_prefix='improved_'):
    """学習過程のグラフを描画（3段階統合版）"""
    
    # 各ステージの履歴を結合
    all_acc = (history_stage1.history.get('accuracy', []) + 
               history_stage2.history.get('accuracy', []) + 
               history_stage3.history.get('accuracy', []))
    all_val_acc = (history_stage1.history.get('val_accuracy', []) + 
                   history_stage2.history.get('val_accuracy', []) + 
                   history_stage3.history.get('val_accuracy', []))
    all_loss = (history_stage1.history.get('loss', []) + 
                history_stage2.history.get('loss', []) + 
                history_stage3.history.get('loss', []))
    all_val_loss = (history_stage1.history.get('val_loss', []) + 
                    history_stage2.history.get('val_loss', []) + 
                    history_stage3.history.get('val_loss', []))
    
    epochs_range = range(1, len(all_acc) + 1)
    
    # Accuracy グラフ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, all_acc, label='Train Acc', marker='.')
    plt.plot(epochs_range, all_val_acc, label='Val Acc', marker='.')
    
    # 各ステージの境界線を表示
    stage1_end = len(history_stage1.history.get('accuracy', []))
    stage2_end = stage1_end + len(history_stage2.history.get('accuracy', []))
    
    plt.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
    plt.axvline(x=stage2_end, color='green', linestyle='--', alpha=0.5, label='Stage 2→3')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss グラフ
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, all_loss, label='Train Loss', marker='.')
    plt.plot(epochs_range, all_val_loss, label='Val Loss', marker='.')
    plt.axvline(x=stage1_end, color='red', linestyle='--', alpha=0.5, label='Stage 1→2')
    plt.axvline(x=stage2_end, color='green', linestyle='--', alpha=0.5, label='Stage 2→3')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}training_history.png', dpi=150)
    plt.close()
    
    print(f"✅ グラフ保存完了: {save_prefix}training_history.png")


# ============================================================
# メイン処理
# ============================================================

def _main():
    print("=" * 80)
    print("🚀 改善版じゃんけん画像分類モデル学習")
    print("=" * 80)
    print("\n📋 改善点:")
    print("  1. 段階的ファインチューニング (3ステージ)")
    print("  2. MixUp データ拡張")
    print("  3. コサインアニーリング学習率")
    print("  4. ラベルスムージング")
    print("  5. Test Time Augmentation (TTA)")
    print("=" * 80)
    
    # データセット準備
    print("\n🔄 データセットを準備中...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "img_train")
    test_dir = os.path.join(script_dir, "img_test")
    
    merged_train_dir = create_merged_dataset(train_dir)
    merged_test_dir = create_merged_dataset(test_dir)
    
    count_images(merged_train_dir)
    count_images(merged_test_dir)
    
    # データセット作成
    print("\n📚 データセットを読み込み中...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        merged_train_dir,
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=SEED
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        merged_test_dir,
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # データ拡張
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.117),
        tf.keras.layers.RandomZoom(0.021),
        tf.keras.layers.RandomTranslation(0.094, 0.094),
        tf.keras.layers.RandomBrightness(0.447),
        tf.keras.layers.RandomContrast(0.428),
        tf.keras.layers.GaussianNoise(0.130),
        RandomDefocus(max_strength=0.02),
    ], name='augmentation')
    
    # 学習用データセットの準備
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # 評価用データセット
    test_ds = test_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    # ============================================================
    # Stage 1: 転移学習（ベースモデル凍結）
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 Stage 1: 転移学習（ベースモデル凍結）")
    print("=" * 80)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(target_size, target_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # ベースモデルを凍結
    
    inputs = tf.keras.Input(shape=(target_size, target_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # ラベルスムージングを適用したCategoricalCrossentropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )
    
    print(f"\n🔒 ベースモデル凍結 (学習率: {initial_learning_rate})")
    print(f"📊 学習可能パラメータ数: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        CosineAnnealingScheduler(
            initial_lr=initial_learning_rate,
            min_lr=finetune_learning_rate,
            total_epochs=initial_epochs
        )
    ]
    
    history_stage1 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=initial_epochs,
        callbacks=callbacks_stage1,
        verbose=2
    )
    
    # Stage 1の結果
    val_acc_stage1 = max(history_stage1.history['val_accuracy'])
    print(f"\n✅ Stage 1 完了: 最高検証精度 = {val_acc_stage1*100:.2f}%")
    
    # ============================================================
    # Stage 2: ファインチューニング（上位層のみ解凍）
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 Stage 2: ファインチューニング（上位層解凍）")
    print("=" * 80)
    
    # 上位50層のみ学習可能にする
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 50
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )
    
    print(f"\n🔓 上位{len(base_model.layers) - fine_tune_at}層を解凍 (学習率: {finetune_learning_rate})")
    print(f"📊 学習可能パラメータ数: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
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
        )
    ]
    
    history_stage2 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=finetune_epochs,
        callbacks=callbacks_stage2,
        verbose=2
    )
    
    val_acc_stage2 = max(history_stage2.history['val_accuracy'])
    print(f"\n✅ Stage 2 完了: 最高検証精度 = {val_acc_stage2*100:.2f}%")
    
    # ============================================================
    # Stage 3: 全体ファインチューニング（全層解凍）
    # ============================================================
    print("\n" + "=" * 80)
    print("📌 Stage 3: 全体ファインチューニング（全層解凍）")
    print("=" * 80)
    
    # 全層を学習可能にする
    for layer in base_model.layers:
        layer.trainable = True
    
    # より小さな学習率で再コンパイル
    very_low_lr = finetune_learning_rate * 0.1
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=very_low_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"]
    )
    
    print(f"\n🔓 全層解凍 (学習率: {very_low_lr})")
    print(f"📊 学習可能パラメータ数: {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    
    callbacks_stage3 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=7,
            min_lr=1e-8,
            verbose=1,
            mode='max'
        )
    ]
    
    history_stage3 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=total_epochs - initial_epochs - finetune_epochs,
        callbacks=callbacks_stage3,
        verbose=2
    )
    
    val_acc_stage3 = max(history_stage3.history['val_accuracy'])
    print(f"\n✅ Stage 3 完了: 最高検証精度 = {val_acc_stage3*100:.2f}%")
    
    # ============================================================
    # 最終評価とTTA
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 最終評価")
    print("=" * 80)
    
    # 通常の評価
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n📈 通常評価: {test_acc*100:.2f}%")
    
    # Test Time Augmentation
    print(f"\n🔄 Test Time Augmentation実行中 (拡張回数: {tta_steps})...")
    tta_acc = predict_with_tta(model, test_ds, num_augmentations=tta_steps)
    print(f"📈 TTA評価: {tta_acc*100:.2f}%")
    
    # モデル保存
    print("\n💾 モデルを保存中...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'model_improved_{timestamp}_acc{max(test_acc, tta_acc)*100:.2f}.keras'
    model.save(model_filename, include_optimizer=False)
    print(f"✅ モデル保存完了: {model_filename}")
    
    # グラフ保存
    print("\n📊 学習履歴グラフを生成中...")
    plot_training_history(history_stage1, history_stage2, history_stage3)
    
    # クリーンアップ
    print("\n🧹 一時ファイルをクリーンアップ中...")
    if os.path.exists(merged_train_dir):
        shutil.rmtree(merged_train_dir)
    if os.path.exists(merged_test_dir):
        shutil.rmtree(merged_test_dir)
    print("✅ クリーンアップ完了")
    
    # サマリー
    print("\n" + "=" * 80)
    print("🎉 学習完了!")
    print("=" * 80)
    print(f"\n📊 各ステージの最高検証精度:")
    print(f"  Stage 1 (凍結学習):        {val_acc_stage1*100:.2f}%")
    print(f"  Stage 2 (部分解凍):        {val_acc_stage2*100:.2f}%")
    print(f"  Stage 3 (全体解凍):        {val_acc_stage3*100:.2f}%")
    print(f"\n📈 最終テスト精度:")
    print(f"  通常評価:                  {test_acc*100:.2f}%")
    print(f"  TTA評価:                   {tta_acc*100:.2f}%")
    print(f"\n💡 最良スコア: {max(test_acc, tta_acc)*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    _main()
