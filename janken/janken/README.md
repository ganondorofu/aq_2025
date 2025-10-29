# 🤖 じゃんけん判定AI - Rock Paper Scissors Classifier

MobileNetV2をベースにした高精度なじゃんけん画像分類システム

## 📊 モデル性能

- **総合精度**: 88.14%
- **クラス別F1スコア**: 
  - ぐー: 90.57%
  - ちょき: 81.25%
  - ぱー: 90.91%

## 🚀 主な特徴

### 1. 遺伝的アルゴリズムによる最適化
`smart_optimization_search.py`を使用してデータ拡張パラメータを最適化:
- **最適化手法**: 遺伝的アルゴリズム + 焼きなまし法
- **実験回数**: 約120回（全探索の1,296通りと比較して大幅に効率化）
- **達成精度**: 89.83% (検証セット)

### 2. 最適化されたデータ拡張パラメータ
```python
rotation: 0.117      # ±42.3°
zoom: 0.021          # ±2.1%
translation: 0.094   # ±9.4%
brightness: 0.447    # ±44.7%
contrast: 0.428      # ±42.8%
noise: 0.130         # 13.0%
```

### 3. ブレ対策の実装
- **RandomDefocus**: ピンボケシミュレーション
- **RandomMotionBlur**: 手ブレシミュレーション（方向性ブレ対応）
- **GaussianNoise**: ノイズ耐性強化

### 4. サブディレクトリ対応
複数のサブディレクトリに分散した画像データを自動統合して学習

## 📁 ファイル構成

```
janken/
├── janken_train_with_subdirs.py    # メイン学習スクリプト
├── janken_predict_aaa.py            # 予測・評価スクリプト
├── smart_optimization_search.py     # GA最適化スクリプト
├── model_with_subdirs.keras         # 学習済みモデル
├── best_model_checkpoint.keras      # ベストチェックポイント
├── img_train/                       # 学習データ
│   ├── 0_gu/
│   ├── 1_tyoki/
│   └── 2_pa/
├── img_test/                        # テストデータ
│   ├── 0_gu/
│   ├── 1_tyoki/
│   └── 2_pa/
└── prediction_report/               # 予測結果レポート
    ├── PREDICTION_REPORT.md
    ├── failed_images/
    └── correct_images/
```

## 🛠️ 使用方法

### 環境構築

```bash
# Conda環境の作成
conda create -n aq2025 python=3.10
conda activate aq2025

# 依存パッケージのインストール
pip install tensorflow keras pillow matplotlib numpy scikit-learn
```

### 学習の実行

```bash
# 基本的な学習
python janken_train_with_subdirs.py

# 遺伝的アルゴリズムで最適化（推奨）
python smart_optimization_search.py
```

### 予測・評価

```bash
python janken_predict_aaa.py
```

予測結果は `prediction_report/PREDICTION_REPORT.md` に詳細レポートが生成されます。

## 🔧 カスタマイズ

### ハイパーパラメータの調整

`janken_train_with_subdirs.py` の以下の変数を変更:

```python
target_size = 224           # 入力画像サイズ
batch_size = 16             # バッチサイズ（GPUメモリに応じて調整）
epochs = 100                # 最大エポック数
learning_rate = 0.0001      # 学習率
MAX_SUBDIR_IMAGES = None    # サブディレクトリから使用する画像数（None=全て）
```

### データ拡張の調整

`data_augmentation` Sequential内のパラメータを変更することで、拡張強度を調整可能。

## 📈 学習結果の詳細

### 混同行列

| 実際＼予測 | ぐー | ちょき | ぱー |
|------------|------|--------|------|
| **ぐー** | 24 (100.0%) | 0 (0.0%) | 0 (0.0%) |
| **ちょき** | 4 (21.1%) | 13 (68.4%) | 2 (10.5%) |
| **ぱー** | 1 (6.2%) | 0 (0.0%) | 15 (93.8%) |

### 主な誤検出パターン

- **ちょき → ぐー**: 4件（特定の角度・照明条件）
- **ちょき → ぱー**: 2件
- **ぱー → ぐー**: 1件

### 改善履歴

1. **初期モデル**: 79.66%
2. **GA最適化適用**: 86.44% (+6.78%)
3. **ブレ対策追加**: 88.14% (+1.7%)

## 🎯 今後の改善方向

1. **データ収集**: 手の甲側・指先側からの画像を追加
2. **ハードネガティブマイニング**: 誤分類画像を重点的に学習
3. **アンサンブル学習**: 複数モデルの組み合わせ
4. **軽量化**: モデルの量子化・プルーニング

## 📚 技術スタック

- **フレームワーク**: TensorFlow / Keras
- **ベースモデル**: MobileNetV2 (ImageNet事前学習済み)
- **最適化手法**: 遺伝的アルゴリズム、焼きなまし法
- **データ拡張**: Keras Preprocessing Layers
- **評価**: scikit-learn (classification_report, confusion_matrix)

## 📄 データセットについて

### 利用データセット

このプロジェクトは、Laurence Moroney（lmoroney@gmail.com, laurencemoroney.com）作成の「Rock Paper Scissors Dataset」を利用しています。本データセットはCC BY 2.0（クリエイティブ・コモンズ 表示 2.0）ライセンスの元で提供されています。

**English:**
This project uses the Rock Paper Scissors Dataset created by Laurence Moroney (lmoroney@gmail.com, laurencemoroney.com), licensed under CC BY 2.0.

### 参考URL

- [データセット配布元（Kaggle）](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
- [公式サイト](http://laurencemoroney.com/)
- [CC BY 2.0 ライセンス詳細](https://creativecommons.org/licenses/by/2.0/)

## 🤝 貢献

プロジェクトへの貢献を歓迎します！以下の方法で貢献できます:

- バグ報告・機能要望: Issueを作成
- コード改善: Pull Requestを送信
- データ追加: 新しい角度・照明条件の画像を提供

## 📝 ライセンス

このプロジェクトのコードはMITライセンスの下で公開されています。  
データセットはCC BY 2.0ライセンスに従います。

## 👤 作成者

**ganondorofu**
- GitHub: [@ganondorofu](https://github.com/ganondorofu)
- Repository: [aq-2025](https://github.com/ganondorofu/aq-2025)

---

**Last Updated**: 2025年10月30日  
**Model Version**: v1.2 (RandomDefocus + GA Optimization)
