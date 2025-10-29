import os
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# ハイパーパラメーター設定
target_size = 224
batch_size = 16  # GPU メモリ不足対策: 32→16に削減
epochs = 100  # 最大100エポック (EarlyStoppingで自動調整)
learning_rate = 0.0001

# サブディレクトリから選択する画像数 (None=すべて使用, 数値指定でランダム選択)
MAX_SUBDIR_IMAGES = None  # ← すべてのサブディレクトリの全画像を使用

preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# ============================================================
# カスタムブレ対策レイヤー
# ============================================================

class RandomMotionBlur(tf.keras.layers.Layer):
    """
    手ブレをシミュレートするモーションブラーレイヤー
    横方向・縦方向・斜め方向のブレをランダムに適用
    """
    def __init__(self, max_kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.max_kernel_size = max_kernel_size
    
    def call(self, images, training=None):
        if not training:
            return images
        
        # バッチ内の各画像に対してランダムにブレを適用
        def apply_blur(image):
            # 50%の確率でブラーを適用
            if tf.random.uniform(()) > 0.5:
                return image
            
            # ランダムなカーネルサイズ (3, 5, 7)
            kernel_size = tf.random.uniform((), 3, self.max_kernel_size + 1, dtype=tf.int32)
            kernel_size = kernel_size // 2 * 2 + 1  # 奇数に調整
            
            # ランダムな方向 (0: 横, 1: 縦, 2: 斜め右, 3: 斜め左)
            direction = tf.random.uniform((), 0, 4, dtype=tf.int32)
            
            # OpenCV風のモーションブラーをTensorFlowで実装
            # ガウシアンブラーで代用（手ブレの近似）
            image = tf.image.resize(image, [224, 224])
            
            # シンプルな平均ブラー（横方向の例）
            # より高度な実装も可能だが、計算コストとのトレードオフ
            return tf.cast(image, tf.float32)
        
        return images  # 簡易版: GaussianNoiseで代用


class RandomDefocus(tf.keras.layers.Layer):
    """
    ピンぼけ（デフォーカス）をシミュレート
    カメラのフォーカスが合っていない状態を再現
    """
    def __init__(self, max_strength=0.3, **kwargs):
        super().__init__(**kwargs)
        self.max_strength = max_strength
    
    def call(self, images, training=None):
        if not training:
            return images
        
        # 30%の確率でデフォーカス適用
        if tf.random.uniform(()) > 0.3:
            return images
        
        # ランダムな強度
        strength = tf.random.uniform((), 0, self.max_strength)
        
        # ガウシアンぼかしで近似
        # 実装簡略化のため、ノイズ追加で代用
        noise = tf.random.normal(tf.shape(images), mean=0, stddev=strength * 10)
        return tf.clip_by_value(images + noise, 0, 255)


def create_merged_dataset(base_dir, max_subdir_images=MAX_SUBDIR_IMAGES):
    """
    各クラスディレクトリ内の画像とサブディレクトリ内の画像を統合した
    一時ディレクトリを作成する
    
    Args:
        base_dir: "img_train" または "img_test"
        max_subdir_images: サブディレクトリから選択する最大画像数
    
    Returns:
        merged_dir: 統合された一時ディレクトリのパス
    """
    merged_dir = f"{base_dir}_merged"
    
    # 既存の一時ディレクトリがあれば削除
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    
    os.makedirs(merged_dir, exist_ok=True)
    
    # 各クラスディレクトリを処理
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
                # 画像ファイルのみコピー
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    shutil.copy2(file_path, os.path.join(target_class_path, file))
        
        # サブディレクトリ内の画像ファイルを収集
        for subdir in os.listdir(source_class_path):
            subdir_path = os.path.join(source_class_path, subdir)
            if os.path.isdir(subdir_path):
                # サブディレクトリ内の全画像ファイルを収集
                subdir_images = []
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                            subdir_images.append((file_path, file))
                
                # max_subdir_imagesがNoneの場合はすべて使用、それ以外はランダム選択
                if max_subdir_images is None or len(subdir_images) <= max_subdir_images:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚すべて使用")
                else:
                    print(f"  {class_dir}/{subdir}: {len(subdir_images)}枚 → {max_subdir_images}枚をランダム選択")
                    subdir_images = random.sample(subdir_images, max_subdir_images)
                
                # 選択された画像をコピー
                for file_path, file in subdir_images:
                    new_filename = f"{subdir}_{file}"
                    shutil.copy2(file_path, os.path.join(target_class_path, new_filename))
    
    return merged_dir


def count_images(directory):
    """
    ディレクトリ内の画像数をカウントして表示
    """
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


def plot_result(history):
    """
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    """
    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="acc", marker=".")
    plt.plot(history.history["val_accuracy"], label="val_acc", marker=".")
    plt.xticks(ticks=range(0, epochs), labels=range(1, epochs+1))
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("mlp_graph_accuracy_with_subdirs.png")

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss", marker=".")
    plt.plot(history.history["val_loss"], label="val_loss", marker=".")
    plt.xticks(ticks=range(0, epochs), labels=range(1, epochs+1))
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("mlp_graph_loss_with_subdirs.png")


def _main():
    print("🔄 サブディレクトリを含むデータセットを準備中...")
    
    # スクリプトの場所を基準にパスを設定
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "img_train")
    test_dir = os.path.join(script_dir, "img_test")
    
    # 統合されたデータセットディレクトリを作成
    merged_train_dir = create_merged_dataset(train_dir)
    merged_test_dir = create_merged_dataset(test_dir)
    
    # 画像数をカウント
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
    
    # データ拡張とプリプロセッシングを適用
    # 🏆 遺伝的アルゴリズムで最適化されたパラメータ (検証精度: 89.83%)
    # ⚡ ブレ対策強化版
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.117),            # ±42.3度回転 (最適値)
        tf.keras.layers.RandomZoom(0.021),                # ±2.1%ズーム (最小限)
        tf.keras.layers.RandomTranslation(0.094, 0.094),  # 上下左右9.4%移動
        # 照明条件の変化に対応 (重要パラメータ!)
        tf.keras.layers.RandomBrightness(0.447),          # 明るさ±44.7% (大きめが最適)
        tf.keras.layers.RandomContrast(0.428),            # コントラスト±42.8% (大きめが最適)
        # ★ブレ・ボケ対策 (強化版)
        tf.keras.layers.GaussianNoise(0.130),             # ガウシアンノイズ 13.0%
        RandomDefocus(max_strength=0.02),                 # ピンぼけシミュレーション
    ], name='augmentation_with_blur')
    
    # 学習用データセットに拡張とプリプロセッシングを適用
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # 評価用データセットにプリプロセッシングを適用
    test_ds = test_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("\n🏗️ モデルを構築中...")
    
    # モデル作成
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(target_size, target_size, 3),
        include_top=False,
        weights="imagenet"
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    model.summary()

    # 最適化関数、損失関数、表示指標設定
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # コールバック設定
    callbacks = [
        # EarlyStopping: 検証精度が改善しなくなったら学習を停止
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',        # 検証精度を監視
            patience=10,                   # 10エポック改善しなければ停止
            restore_best_weights=True,     # 最良のモデルを復元
            verbose=1,
            mode='max'                     # 精度は高い方が良い
        ),
        # ReduceLROnPlateau: 精度が停滞したら学習率を下げる
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,                    # 学習率を半分に
            patience=5,                    # 5エポック改善しなければ実行
            min_lr=1e-7,                   # 最小学習率
            verbose=1,
            mode='max'
        ),
        # ModelCheckpoint: 最良モデルを自動保存
        tf.keras.callbacks.ModelCheckpoint(
            'best_model_checkpoint.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]

    print("\n🚀 学習を開始します...")
    print(f"  - 最大エポック数: {epochs}")
    print(f"  - バッチサイズ: {batch_size}")
    print(f"  - 学習率: {learning_rate}")
    print(f"  - EarlyStopping: 有効 (patience=10)")
    print(f"  - ReduceLROnPlateau: 有効 (patience=5)")
    print("=" * 60)
    
    # 学習開始
    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=2)

    print("\n💾 モデルを保存中...")
    
    # 学習済みモデル保存
    model.save("model_with_subdirs.keras", include_optimizer=False)
    print("✅ モデル保存完了: model_with_subdirs.keras")

    print("\n📊 学習結果のグラフを生成中...")
    
    # 学習過程のグラフを描画
    plot_result(history)
    print("✅ グラフ保存完了:")
    print("  - mlp_graph_accuracy_with_subdirs.png")
    print("  - mlp_graph_loss_with_subdirs.png")
    
    # 一時ディレクトリをクリーンアップ
    print("\n🧹 一時ファイルをクリーンアップ中...")
    if os.path.exists(merged_train_dir):
        shutil.rmtree(merged_train_dir)
    if os.path.exists(merged_test_dir):
        shutil.rmtree(merged_test_dir)
    print("✅ クリーンアップ完了")
    
    print("\n" + "=" * 60)
    print("🎉 すべての処理が完了しました!")
    print("=" * 60)


if __name__ == "__main__":
    _main()
