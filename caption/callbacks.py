from keras.callbacks import Callback


class CaptionCallback(Callback):
    def __init__(self, model, monitor='val_loss'):
        super().__init__()
        self._model = model
        self._monitor = monitor
        self._best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if logs[self._monitor] < self._best:
            self._best = logs[self._monitor]
            self._model.save()
        else:
            print('\nNo improvement on model, skipping save...')
