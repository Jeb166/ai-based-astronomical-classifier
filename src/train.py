"""python -m src.train --csv data/skyserver.csv --epochs 60"""

import argparse, tensorflow as tf, os
from datamodule import load_data
from models      import build_wide_bn

def main(args):
    (Xt,yt),(Xv,yv),(Xte,yte),f = load_data(args.csv)

    yt  = tf.keras.utils.to_categorical(yt,  3)
    yv  = tf.keras.utils.to_categorical(yv,  3)

    model = build_wide_bn(f)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(args.lr),
                  metrics=["categorical_accuracy"])

    ckpt_path = os.path.join("outputs",
        f"widebn_best_{{val_categorical_accuracy:.3f}}.h5")
    os.makedirs("outputs", exist_ok=True)

    cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
            monitor="val_categorical_accuracy",
            save_best_only=True, mode="max", verbose=1)

    model.fit(Xt, yt,
              validation_data=(Xv,yv),
              epochs=args.epochs,
              batch_size=args.bs,
              callbacks=[cb])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",    default="data/skyserver.csv")
    p.add_argument("--epochs", type=int,   default=60)
    p.add_argument("--bs",     type=int,   default=512)
    p.add_argument("--lr",     type=float, default=1e-3)
    main(p.parse_args())
