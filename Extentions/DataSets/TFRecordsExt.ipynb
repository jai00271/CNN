{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "TFRecordsExt.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3Kd18EK1QVWb",
        "colab_type": "text"
      },
      "source": [
        ""
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gxcbGNDDmEaB",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "class TFRecords:   \n",
        "  def __init__(self, out_filename, images, labels):\n",
        "    self.out_filename = out_filename\n",
        "    self.images = images\n",
        "    self.labels = labels\n",
        "\n",
        "  def _int64_feature(self, value):\n",
        "    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))\n",
        "\n",
        "  def _bytes_feature(self, value):\n",
        "    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))\n",
        "    \n",
        "  def createDataRecord(self, out_filename, images, labels):\n",
        "\n",
        "    writer = tf.io.TFRecordWriter(out_filename)\n",
        "\n",
        "    for i in range(len(images)):\n",
        "      feature = {\n",
        "          'image_raw': self._bytes_feature(images[i].tostring()),\n",
        "          'label': self._int64_feature(labels[i])\n",
        "      }\n",
        "\n",
        "      example = tf.train.Example(features=tf.train.Features(feature=feature))\n",
        "\n",
        "      writer.write(example.SerializeToString())\n",
        "\n",
        "    writer.close()\n",
        "    sys.stdout.flush()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_UMIl-q5SawS",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}