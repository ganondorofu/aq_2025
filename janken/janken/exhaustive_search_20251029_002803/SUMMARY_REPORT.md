# 🔬 データ拡張パラメータ徹底サーチ結果

**実行日時**: 2025年10月29日 08:46:37

**総実験数**: 150件

**総学習時間**: 29523.5秒 (8.20時間)

---

## 🏆 TOP 10 最高精度の設定

| 順位 | 実験名 | 検証精度 | 学習精度 | エポック数 |
|------|--------|----------|----------|------------|
| 68 | trans10pct_bright20pct_cont30pct_noise5pct | 0.8983 | 1.0000 | 25 |
| 27 | bright30pct_noise8pct | 0.8814 | 1.0000 | 26 |
| 80 | trans10pct_bright30pct_cont30pct_noise5pct | 0.8644 | 1.0000 | 21 |
| 140 | trans15pct_bright40pct_cont30pct_noise5pct | 0.8475 | 1.0000 | 25 |
| 130 | trans15pct_bright30pct_cont40pct | 0.8475 | 1.0000 | 21 |
| 148 | zoom15pct_cont20pct | 0.8305 | 1.0000 | 30 |
| 92 | trans10pct_bright40pct_cont30pct_noise5pct | 0.8305 | 1.0000 | 30 |
| 134 | trans15pct_bright40pct_noise5pct | 0.8305 | 1.0000 | 14 |
| 99 | trans15pct_noise8pct | 0.8136 | 1.0000 | 30 |
| 75 | trans10pct_bright30pct_noise8pct | 0.8136 | 1.0000 | 25 |

---

## 📊 パラメータ別の影響分析

### Rotation

```
             mean       max  count
rotation                          
0.0       0.64565  0.898305    150
```

### Zoom

```
          mean       max  count
zoom                           
0.00  0.647952  0.898305    144
0.15  0.590395  0.830508      6
```

### Translation

```
                 mean       max  count
translation                           
0.00         0.635907  0.881356     54
0.10         0.662076  0.898305     48
0.15         0.640184  0.847458     48
```

### Brightness

```
                mean       max  count
brightness                           
0.0         0.629136  0.830508     42
0.2         0.658192  0.898305     36
0.3         0.648305  0.881356     36
0.4         0.649718  0.847458     36
```

### Contrast

```
              mean       max  count
contrast                           
0.0       0.651456  0.881356     39
0.2       0.612342  0.830508     39
0.3       0.663371  0.898305     36
0.4       0.657721  0.847458     36
```

### Noise

```
           mean       max  count
noise                           
0.00   0.642712  0.847458     50
0.05   0.647119  0.898305     50
0.08   0.647119  0.881356     50
```

---

## 💡 最適パラメータの推奨値

**最高精度**: 0.8983

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.GaussianNoise(0.05),
])
```

