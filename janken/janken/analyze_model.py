"""
高精度モデルの構造と設定を詳細に分析するスクリプト
"""
import tensorflow as tf
import os
from datetime import datetime

def analyze_model(model_path):
    """
    モデルファイルから可能な限り情報を抽出
    """
    print("=" * 80)
    print(f"🔍 モデル分析: {model_path}")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ エラー: {model_path} が見つかりません")
        return
    
    # ファイル情報
    file_stat = os.stat(model_path)
    file_size = file_stat.st_size / (1024 * 1024)  # MB
    created_time = datetime.fromtimestamp(file_stat.st_ctime)
    modified_time = datetime.fromtimestamp(file_stat.st_mtime)
    
    print(f"\n📁 ファイル情報:")
    print(f"  サイズ: {file_size:.2f} MB")
    print(f"  作成日時: {created_time}")
    print(f"  更新日時: {modified_time}")
    
    # モデル読み込み
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"\n✅ モデル読み込み成功")
    except Exception as e:
        print(f"\n❌ モデル読み込みエラー: {e}")
        return
    
    # モデル構造
    print(f"\n🏗️ モデル構造:")
    print(f"  入力shape: {model.input_shape}")
    print(f"  出力shape: {model.output_shape}")
    print(f"  総レイヤー数: {len(model.layers)}")
    print(f"  総パラメータ数: {model.count_params():,}")
    
    # 訓練可能パラメータ
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"  訓練可能: {trainable_count:,}")
    print(f"  固定: {non_trainable_count:,}")
    
    # ベースモデルの検出
    print(f"\n🔎 使用されているベースモデル:")
    base_model_detected = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model_detected = "MobileNet"
        elif 'efficientnet' in layer.name.lower():
            base_model_detected = "EfficientNet"
        elif 'resnet' in layer.name.lower():
            base_model_detected = "ResNet"
        elif 'vgg' in layer.name.lower():
            base_model_detected = "VGG"
    
    if base_model_detected:
        print(f"  ✅ 検出: {base_model_detected}")
    else:
        print(f"  ⚠️ 不明（カスタムモデルの可能性）")
    
    # データ拡張レイヤーの検出
    print(f"\n🎨 データ拡張レイヤー:")
    augmentation_layers = []
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        if any(keyword in layer_type.lower() for keyword in 
               ['random', 'augment', 'flip', 'rotation', 'zoom', 'translation', 
                'brightness', 'contrast', 'noise', 'defocus', 'blur']):
            augmentation_layers.append((i, layer.name, layer_type))
            print(f"  [{i}] {layer.name} ({layer_type})")
            
            # パラメータを抽出（可能な場合）
            config = layer.get_config()
            for key, value in config.items():
                if key not in ['name', 'trainable', 'dtype']:
                    print(f"      {key}: {value}")
    
    if not augmentation_layers:
        print(f"  ⚠️ データ拡張レイヤーが見つかりません")
        print(f"  （データ拡張は学習時に別途適用された可能性があります）")
    
    # 全レイヤーの詳細
    print(f"\n📋 全レイヤー詳細:")
    print("-" * 80)
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        output_shape = layer.output_shape
        params = layer.count_params()
        trainable = "訓練可能" if layer.trainable else "固定"
        
        print(f"[{i:3d}] {layer.name:30s} {layer_type:25s} {str(output_shape):30s} {params:>10,} {trainable}")
    
    print("-" * 80)
    
    # オプティマイザ情報（保存されている場合）
    print(f"\n⚙️ コンパイル情報:")
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        print(f"  オプティマイザ: {model.optimizer.__class__.__name__}")
        if hasattr(model.optimizer, 'learning_rate'):
            lr = model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                print(f"  学習率: {lr.numpy()}")
            else:
                print(f"  学習率: {lr}")
    else:
        print(f"  ⚠️ オプティマイザ情報なし（include_optimizer=Falseで保存された可能性）")
    
    print("\n" + "=" * 80)
    print("✅ 分析完了")
    print("=" * 80)


def compare_models(model_paths):
    """
    複数のモデルを比較
    """
    print("\n" + "=" * 80)
    print("📊 モデル比較")
    print("=" * 80)
    
    for path in model_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            params = model.count_params()
            print(f"\n{path}:")
            print(f"  パラメータ数: {params:,}")
            print(f"  レイヤー数: {len(model.layers)}")


if __name__ == "__main__":
    print("🔍 モデル分析ツール")
    print("=" * 80)
    
    # 現在のディレクトリ内の.kerasファイルを検索（.bakも含む）
    keras_files = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('.keras.bak')]
    
    if not keras_files:
        print("❌ .kerasファイルが見つかりません")
    else:
        print(f"\n📁 見つかったモデルファイル ({len(keras_files)}個):")
        for i, f in enumerate(keras_files, 1):
            file_stat = os.stat(f)
            file_size = file_stat.st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(file_stat.st_mtime)
            print(f"  {i}. {f:40s} {file_size:8.2f} MB  {modified_time}")
        
        print("\n" + "=" * 80)
        choice = input("分析するモデルの番号を入力 (0=すべて): ")
        
        if choice == "0":
            for f in keras_files:
                analyze_model(f)
                print("\n")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(keras_files):
                    analyze_model(keras_files[idx])
                else:
                    print("❌ 無効な番号です")
            except ValueError:
                print("❌ 数字を入力してください")
