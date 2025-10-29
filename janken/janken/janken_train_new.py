import os
import tensorflow as tf
import matplotlib.pyplot as plt


# ハイパーパラメーター設定
target_size = 224
batch_size = 32
epochs = 20
learning_rate = 0.0001

preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


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
    plt.savefig("mlp_graph_accuracy.png")

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss", marker=".")
    plt.plot(history.history["val_loss"], label="val_loss", marker=".")
    plt.xticks(ticks=range(0, epochs), labels=range(1, epochs+1))
    plt.grid()
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("mlp_graph_loss.png")


def _main():
    # 学習用データセット作成
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "img_train",
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    # 評価用データセット作成
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "img_test",
        image_size=(target_size, target_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # データ拡張とプリプロセッシングを適用
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(15.0/360.0),
        tf.keras.layers.RandomFlip("horizontal"),
    ])
    
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

    # 学習開始
    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=epochs,
                        verbose=2)

    # 学習済みモデル保存
    model.save("model.keras", include_optimizer=False)

    # 学習過程のグラフを描画
    plot_result(history)


if __name__ == "__main__":
    _main()