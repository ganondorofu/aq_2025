"""
データ拡張パラメータの最適化
焼きなまし法 + 遺伝的アルゴリズムのハイブリッド探索
"""

import os
import shutil
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json
import math


# ハイパーパラメーター
target_size = 224
batch_size = 16  # GPU メモリ不足対策: 32→16に削減
epochs = 15  # 高速評価のため削減
learning_rate = 0.0001

# 最適化パラメータ
POPULATION_SIZE = 8          # 遺伝的アルゴリズムの集団サイズ
GENERATIONS = 15             # 世代数
SIMULATED_ANNEALING_TEMP = 1.0  # 焼きなまし法の初期温度
COOLING_RATE = 0.9           # 冷却率
ELITE_SIZE = 2               # エリート選択数

preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
optimizer_class = tf.keras.optimizers.Adam


class AugmentationParams:
    """データ拡張パラメータクラス"""
    
    def __init__(self, rotation=0.25, zoom=0.15, translation=0.1, 
                 brightness=0.3, contrast=0.3, noise=0.05):
        self.rotation = rotation
        self.zoom = zoom
        self.translation = translation
        self.brightness = brightness
        self.contrast = contrast
        self.noise = noise
        self.fitness = 0.0  # 適応度（検証精度）
        
    def to_dict(self):
        return {
            'rotation': self.rotation,
            'zoom': self.zoom,
            'translation': self.translation,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'noise': self.noise,
            'fitness': self.fitness
        }
    
    def mutate(self, temperature=1.0):
        """突然変異 (焼きなまし法の温度パラメータ付き)"""
        params = AugmentationParams(
            rotation=self.rotation,
            zoom=self.zoom,
            translation=self.translation,
            brightness=self.brightness,
            contrast=self.contrast,
            noise=self.noise
        )
        
        # 温度が高いほど大きな変化
        mutation_strength = 0.1 * temperature
        
        # ランダムに1-3個のパラメータを変更
        num_mutations = random.randint(1, 3)
        param_names = ['rotation', 'zoom', 'translation', 'brightness', 'contrast', 'noise']
        
        for _ in range(num_mutations):
            param = random.choice(param_names)
            current_value = getattr(params, param)
            
            # ガウス分布で変更
            delta = np.random.normal(0, mutation_strength)
            new_value = current_value + delta
            
            # 範囲制限
            if param == 'rotation':
                new_value = np.clip(new_value, 0.0, 0.5)  # 0-180度
            elif param == 'zoom':
                new_value = np.clip(new_value, 0.0, 0.4)  # 0-40%
            elif param == 'translation':
                new_value = np.clip(new_value, 0.0, 0.3)  # 0-30%
            elif param in ['brightness', 'contrast']:
                new_value = np.clip(new_value, 0.0, 0.6)  # 0-60%
            elif param == 'noise':
                new_value = np.clip(new_value, 0.0, 0.15)  # 0-15%
            
            setattr(params, param, new_value)
        
        return params
    
    @staticmethod
    def crossover(parent1, parent2):
        """交叉（2点交叉）"""
        child = AugmentationParams()
        
        # ランダムに各パラメータを親から選択
        for param in ['rotation', 'zoom', 'translation', 'brightness', 'contrast', 'noise']:
            if random.random() < 0.5:
                setattr(child, param, getattr(parent1, param))
            else:
                setattr(child, param, getattr(parent2, param))
        
        return child
    
    @staticmethod
    def random():
        """ランダムなパラメータ生成"""
        return AugmentationParams(
            rotation=random.uniform(0.0, 0.5),
            zoom=random.uniform(0.0, 0.4),
            translation=random.uniform(0.0, 0.3),
            brightness=random.uniform(0.0, 0.6),
            contrast=random.uniform(0.0, 0.6),
            noise=random.uniform(0.0, 0.15)
        )


def create_data_augmentation(params):
    """パラメータからデータ拡張レイヤーを作成"""
    layers = []
    
    if params.rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(params.rotation))
    if params.zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(params.zoom))
    if params.translation > 0:
        layers.append(tf.keras.layers.RandomTranslation(params.translation, params.translation))
    if params.brightness > 0:
        layers.append(tf.keras.layers.RandomBrightness(params.brightness))
    if params.contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(params.contrast))
    if params.noise > 0:
        layers.append(tf.keras.layers.GaussianNoise(params.noise))
    
    return tf.keras.Sequential(layers) if layers else None


def evaluate_params(params, train_ds, test_ds, experiment_num, total_experiments):
    """パラメータを評価"""
    print(f"\n{'='*70}")
    print(f"🧬 実験 {experiment_num}/{total_experiments}")
    print(f"{'='*70}")
    print(f"  Rotation:     {params.rotation:.3f} (±{params.rotation*360:.1f}°)")
    print(f"  Zoom:         {params.zoom:.3f} (±{params.zoom*100:.1f}%)")
    print(f"  Translation:  {params.translation:.3f} (±{params.translation*100:.1f}%)")
    print(f"  Brightness:   {params.brightness:.3f} (±{params.brightness*100:.1f}%)")
    print(f"  Contrast:     {params.contrast:.3f} (±{params.contrast*100:.1f}%)")
    print(f"  Noise:        {params.noise:.3f}")
    print(f"{'='*70}\n")
    
    # データ拡張レイヤー作成
    data_augmentation = create_data_augmentation(params)
    
    # データセット準備
    train_dataset = train_ds
    if data_augmentation:
        train_dataset = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    train_dataset = train_dataset.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = test_ds.map(
        lambda x, y: (preprocessing_function(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    # モデル構築
    tf.keras.backend.clear_session()
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(target_size, target_size, 3),
        include_top=False,
        weights="imagenet"
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    
    model.compile(
        optimizer=optimizer_class(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # EarlyStopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
    ]
    
    # 学習
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    # 最高精度を取得
    best_val_accuracy = max(history.history['val_accuracy'])
    params.fitness = best_val_accuracy
    
    print(f"✅ 検証精度: {best_val_accuracy*100:.2f}%\n")
    
    return best_val_accuracy


def simulated_annealing_genetic_algorithm(train_ds, test_ds, output_dir):
    """焼きなまし法 + 遺伝的アルゴリズムのハイブリッド最適化"""
    
    print("\n" + "="*70)
    print("🔥 焼きなまし法 + 遺伝的アルゴリズム ハイブリッド最適化")
    print("="*70)
    print(f"集団サイズ: {POPULATION_SIZE}")
    print(f"世代数: {GENERATIONS}")
    print(f"初期温度: {SIMULATED_ANNEALING_TEMP}")
    print(f"冷却率: {COOLING_RATE}")
    print("="*70 + "\n")
    
    # 初期集団をランダム生成
    population = [AugmentationParams.random() for _ in range(POPULATION_SIZE)]
    
    # ベースライン評価（拡張なし）
    print("\n📊 ベースライン評価（データ拡張なし）")
    baseline = AugmentationParams(0, 0, 0, 0, 0, 0)
    evaluate_params(baseline, train_ds, test_ds, 0, POPULATION_SIZE * GENERATIONS)
    
    best_overall = baseline
    history = []
    experiment_count = 1
    
    # 世代ループ
    for generation in range(GENERATIONS):
        print(f"\n{'#'*70}")
        print(f"🧬 第 {generation+1}/{GENERATIONS} 世代")
        print(f"{'#'*70}\n")
        
        # 現在の温度（焼きなまし法）
        temperature = SIMULATED_ANNEALING_TEMP * (COOLING_RATE ** generation)
        print(f"🌡️ 現在の温度: {temperature:.3f}\n")
        
        # 全個体を評価
        for individual in population:
            if individual.fitness == 0.0:  # 未評価の個体のみ
                evaluate_params(individual, train_ds, test_ds, 
                              experiment_count, POPULATION_SIZE * GENERATIONS)
                experiment_count += 1
        
        # 適応度でソート
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 最良個体の更新
        if population[0].fitness > best_overall.fitness:
            best_overall = population[0]
            print(f"\n🎉 新記録! 検証精度: {best_overall.fitness*100:.2f}%")
        
        # 世代統計
        avg_fitness = np.mean([ind.fitness for ind in population])
        history.append({
            'generation': generation + 1,
            'best_fitness': population[0].fitness,
            'avg_fitness': avg_fitness,
            'temperature': temperature,
            'best_params': population[0].to_dict()
        })
        
        print(f"\n📈 第{generation+1}世代の結果:")
        print(f"  最良: {population[0].fitness*100:.2f}%")
        print(f"  平均: {avg_fitness*100:.2f}%")
        print(f"  最悪: {population[-1].fitness*100:.2f}%")
        
        # 最終世代なら終了
        if generation == GENERATIONS - 1:
            break
        
        # 次世代の生成
        new_population = []
        
        # エリート選択
        new_population.extend(population[:ELITE_SIZE])
        
        # 残りを交叉と突然変異で生成
        while len(new_population) < POPULATION_SIZE:
            # トーナメント選択で親を選ぶ
            parent1 = max(random.sample(population, 3), key=lambda x: x.fitness)
            parent2 = max(random.sample(population, 3), key=lambda x: x.fitness)
            
            # 交叉
            if random.random() < 0.7:  # 70%の確率で交叉
                child = AugmentationParams.crossover(parent1, parent2)
            else:
                child = parent1
            
            # 突然変異（温度に応じた変異）
            if random.random() < 0.8:  # 80%の確率で突然変異
                child = child.mutate(temperature)
            
            new_population.append(child)
        
        population = new_population
    
    # 結果保存
    results = {
        'best_params': best_overall.to_dict(),
        'baseline': baseline.to_dict(),
        'history': history,
        'total_experiments': experiment_count - 1
    }
    
    with open(os.path.join(output_dir, 'optimization_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # グラフ作成
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] * 100 for h in history]
    avg_fitness = [h['avg_fitness'] * 100 for h in history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(generations, best_fitness, 'b-o', label='最良個体', linewidth=2)
    plt.plot(generations, avg_fitness, 'r--s', label='平均', linewidth=2)
    plt.axhline(y=baseline.fitness*100, color='g', linestyle=':', label='ベースライン', linewidth=2)
    plt.xlabel('世代', fontsize=12)
    plt.ylabel('検証精度 (%)', fontsize=12)
    plt.title('遺伝的アルゴリズムによる最適化の進化', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_progress.png'), dpi=150)
    plt.close()
    
    return best_overall


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"smart_optimization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 スマート最適化探索システム")
    print("="*70)
    print(f"出力ディレクトリ: {output_dir}")
    print("="*70 + "\n")
    
    # データセット読み込み
    print("📚 データセットを読み込み中...\n")
    
    # img_trainとimg_testを直接読み込み
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "img_train",
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "img_test",
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # 最適化実行
    best_params = simulated_annealing_genetic_algorithm(train_ds, test_ds, output_dir)
    
    # 最終結果
    print("\n" + "="*70)
    print("🏆 最適化完了!")
    print("="*70)
    print(f"\n最良のパラメータ:")
    print(f"  Rotation:     {best_params.rotation:.3f} (±{best_params.rotation*360:.1f}°)")
    print(f"  Zoom:         {best_params.zoom:.3f} (±{best_params.zoom*100:.1f}%)")
    print(f"  Translation:  {best_params.translation:.3f} (±{best_params.translation*100:.1f}%)")
    print(f"  Brightness:   {best_params.brightness:.3f} (±{best_params.brightness*100:.1f}%)")
    print(f"  Contrast:     {best_params.contrast:.3f} (±{best_params.contrast*100:.1f}%)")
    print(f"  Noise:        {best_params.noise:.3f}")
    print(f"\n検証精度: {best_params.fitness*100:.2f}%")
    print(f"\n結果は {output_dir}/ に保存されました")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
