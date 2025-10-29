"""
複数のモデルアーキテクチャを比較して最適なモデルを見つける
MobileNetV2, EfficientNetB0, EfficientNetV2, ResNet50V2などを比較
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import matplotlib.pyplot as plt

# 設定
TARGET_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50  # 比較用なので短めに
LEARNING_RATE = 0.0001

# テストするモデルの定義
MODELS_TO_TEST = {
    'MobileNetV2': {
        'model': tf.keras.applications.MobileNetV2,
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
        'size': 224
    },
    'MobileNetV3Large': {
        'model': tf.keras.applications.MobileNetV3Large,
        'preprocess': tf.keras.applications.mobilenet_v3.preprocess_input,
        'size': 224
    },
    'EfficientNetB0': {
        'model': tf.keras.applications.EfficientNetB0,
        'preprocess': tf.keras.applications.efficientnet.preprocess_input,
        'size': 224
    },
    'EfficientNetV2B0': {
        'model': tf.keras.applications.EfficientNetV2B0,
        'preprocess': tf.keras.applications.efficientnet_v2.preprocess_input,
        'size': 224
    },
    'ResNet50V2': {
        'model': tf.keras.applications.ResNet50V2,
        'preprocess': tf.keras.applications.resnet_v2.preprocess_input,
        'size': 224
    },
}

# GA最適化済みデータ拡張パラメータ
def create_data_augmentation():
    """遺伝的アルゴリズムで最適化されたデータ拡張"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.117),
        tf.keras.layers.RandomZoom(0.021),
        tf.keras.layers.RandomTranslation(0.094, 0.094),
        tf.keras.layers.RandomBrightness(0.447),
        tf.keras.layers.RandomContrast(0.428),
        tf.keras.layers.GaussianNoise(0.130),
    ], name='optimized_augmentation')


def create_model(model_config, input_size):
    """
    指定されたアーキテクチャでモデルを作成
    """
    base_model = model_config['model'](
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(3, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate(model_name, model_config, train_ds, test_ds):
    """
    モデルを学習して評価
    """
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} の学習を開始")
    print(f"{'='*60}")
    
    # モデル作成
    model = create_model(model_config, model_config['size'])
    
    # パラメータ数を表示
    total_params = model.count_params()
    print(f"📊 総パラメータ数: {total_params:,}")
    
    # コールバック
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 学習開始
    start_time = datetime.now()
    
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # 評価
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    
    # 結果を返す
    return {
        'model_name': model_name,
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_epoch': int(np.argmax(history.history['val_accuracy']) + 1),
        'total_epochs': len(history.history['val_accuracy']),
        'training_time_seconds': training_time,
        'total_params': total_params,
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
    }


def load_datasets(train_dir, test_dir, preprocess_fn):
    """
    データセットをロード
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # データ拡張
    data_augmentation = create_data_augmentation()
    
    # 学習データに拡張を適用
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # テストデータに前処理のみ
    test_ds = test_ds.map(
        lambda x, y: (preprocess_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds


def generate_comparison_report(results, output_dir):
    """
    比較レポートを生成
    """
    # 結果をソート（精度順）
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    
    # Markdown生成
    md_content = f"""# 🏆 モデルアーキテクチャ比較レポート

**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

---

## 📊 総合ランキング

| 順位 | モデル | テスト精度 | 検証精度(最高) | 学習時間 | パラメータ数 |
|------|--------|-----------|---------------|---------|-------------|
"""
    
    for i, result in enumerate(sorted_results, 1):
        md_content += f"| {i} | **{result['model_name']}** | {result['test_accuracy']*100:.2f}% | {result['best_val_accuracy']*100:.2f}% | {result['training_time_seconds']:.0f}秒 | {result['total_params']:,} |\n"
    
    md_content += f"""
---

## 📈 詳細結果

"""
    
    for result in sorted_results:
        md_content += f"""
### {result['model_name']}

- **テスト精度**: {result['test_accuracy']*100:.2f}%
- **検証精度(最高)**: {result['best_val_accuracy']*100:.2f}%
- **到達エポック**: {result['best_epoch']}/{result['total_epochs']}
- **学習時間**: {result['training_time_seconds']:.0f}秒 ({result['training_time_seconds']/60:.1f}分)
- **パラメータ数**: {result['total_params']:,}
- **1エポックあたりの時間**: {result['training_time_seconds']/result['total_epochs']:.1f}秒

"""
    
    md_content += """
---

## 💡 推奨モデル

"""
    
    best_model = sorted_results[0]
    md_content += f"""
**{best_model['model_name']}** が最高精度を達成しました!

- テスト精度: **{best_model['test_accuracy']*100:.2f}%**
- 学習時間: {best_model['training_time_seconds']/60:.1f}分

### 使用方法

```python
base_model = tf.keras.applications.{best_model['model_name']}(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
```

---

*このレポートは `model_comparison.py` により自動生成されました*
"""
    
    # ファイルに保存
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\n✅ レポートを生成: {report_path}")
    
    # グラフ生成
    generate_comparison_graphs(results, output_dir)


def generate_comparison_graphs(results, output_dir):
    """
    比較グラフを生成
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. テスト精度の比較
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    accuracies = [r['test_accuracy'] * 100 for r in results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = ax1.barh(models, accuracies, color=colors)
    ax1.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Model Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontsize=10)
    
    # 2. 学習時間の比較
    ax2 = axes[0, 1]
    times = [r['training_time_seconds']/60 for r in results]
    bars = ax2.barh(models, times, color=colors)
    ax2.set_xlabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, time in zip(bars, times):
        ax2.text(time + 0.5, bar.get_y() + bar.get_height()/2,
                f'{time:.1f}m', va='center', fontsize=10)
    
    # 3. パラメータ数の比較
    ax3 = axes[1, 0]
    params = [r['total_params']/1e6 for r in results]  # Million parameters
    bars = ax3.barh(models, params, color=colors)
    ax3.set_xlabel('Parameters (Millions)', fontsize=12)
    ax3.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    for bar, param in zip(bars, params):
        ax3.text(param + 0.1, bar.get_y() + bar.get_height()/2,
                f'{param:.2f}M', va='center', fontsize=10)
    
    # 4. 精度 vs 学習時間のトレードオフ
    ax4 = axes[1, 1]
    scatter = ax4.scatter(times, accuracies, s=[p*50 for p in params], 
                         c=range(len(models)), cmap='viridis', alpha=0.6)
    
    for i, (time, acc, model) in enumerate(zip(times, accuracies, models)):
        ax4.annotate(model, (time, acc), fontsize=9, 
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_xlabel('Training Time (minutes)', fontsize=12)
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax4.set_title('Accuracy vs Training Time\n(Bubble size = Parameters)', 
                 fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    graph_path = os.path.join(output_dir, 'model_comparison_graphs.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"✅ グラフを生成: {graph_path}")
    plt.close()


def main():
    """メイン処理"""
    print("="*60)
    print("🔬 モデルアーキテクチャ比較実験")
    print("="*60)
    
    # 出力ディレクトリ
    output_dir = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # データディレクトリ（スクリプトの場所から相対パス）
    script_dir = Path(__file__).parent
    train_dir = script_dir / "img_train_merged"
    test_dir = script_dir / "img_test_merged"
    
    # マージされたデータセットが存在しない場合は作成を促す
    if not train_dir.exists():
        print("\n⚠️ img_train_merged が見つかりません")
        print("先に janken_train_with_subdirs.py を実行してデータセットを準備してください")
        return
    
    results = []
    
    # 各モデルを学習・評価
    for model_name, model_config in MODELS_TO_TEST.items():
        print(f"\n{'='*60}")
        print(f"📦 {model_name} をテスト中...")
        print(f"{'='*60}")
        
        try:
            # データセットロード
            train_ds, test_ds = load_datasets(
                str(train_dir), 
                str(test_dir), 
                model_config['preprocess']
            )
            
            # 学習・評価
            result = train_and_evaluate(model_name, model_config, train_ds, test_ds)
            results.append(result)
            
            # 途中経過を保存
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"❌ {model_name} でエラー: {e}")
            continue
    
    # 比較レポート生成
    if results:
        generate_comparison_report(results, output_dir)
        
        print(f"\n{'='*60}")
        print("🎉 すべてのモデルの比較が完了しました!")
        print(f"{'='*60}")
        print(f"\n📁 結果: {output_dir}/")
        print(f"  - model_comparison_report.md")
        print(f"  - model_comparison_graphs.png")
        print(f"  - results.json")
    else:
        print("\n❌ 評価可能なモデルがありませんでした")


if __name__ == "__main__":
    main()
