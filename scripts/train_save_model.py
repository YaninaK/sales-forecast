#!/usr/bin/env python3
"""Train and save model for RecSys-retail"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import argparse
import pandas as pd
import tensorflow as tf
from typing import Optional

from src.sales_forecast.data.make_dataset import load_data
from src.sales_forecast.models import train
from src.sales_forecast.data.train_test_datasets import get_train_dataset
from src.sales_forecast.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d1",
        "--data_path",
        required=False,
        default="01_raw/train.parquet.gzip",
        help="train dataset store path",
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="filename to store model",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")
    data = load_data(args.data_path)

    logging.info("Preprocessing data...")
    X = train.data_preprocessing_pipeline(data)

    logging.info("Training the model...")
    train_store(X, args.output)


def train_store(model, X, filename: str, seed=25):
    """
    Trains and stores LSTM model.
    """
    tf.random.set_seed(seed)

    X_train, y_train = get_train_dataset(X)

    n_shops = X_train.shape[1]
    n_epochs = 10
    batch_size = 80
    m = 4 * n_shops

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 3e-2 * 0.95**epoch
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    model.fit(
        X_train[:-m, :, :],
        y_train[:-m, :, :],
        epochs=n_epochs,
        validation_data=(X_train[-m:, :, :], y_train[-m:, :, :]),
        batch_size=batch_size,
        verbose=1,
        callbacks=[reduce_lr],
    )
    store(model, filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
