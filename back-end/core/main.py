from core import process, predict


def c_main(path, mode, ext):
    image_data = process.pre_process(path)
    predict.predict(image_data, mode, ext)

    return image_data[1] + '.' + ext


if __name__ == '__main__':
    pass
