from keras.callbacks import Callback


class CaptionCallback(Callback):
    def __init__(self, model, samples=None, monitor='val_loss'):
        super().__init__()
        self._model = model
        self._monitor = monitor
        self._best = float('inf')
        self._samples = samples

    def on_epoch_end(self, epoch, logs=None):
        print()
        if self._samples is not None:
            print('Sample captions:')
            for s in self._samples:
                c = self._model.caption(s)
                print(' %s' % c)

        if logs[self._monitor] < self._best:
            self._best = logs[self._monitor]
            self._model.save()
            print('Saved model to %s\n' % self._model.save_dir)
        else:
            print('No improvement on model, skipping save...\n')
