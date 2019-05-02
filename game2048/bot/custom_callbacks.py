import warnings

import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K


class NextBestModelCheckpoint(ModelCheckpoint):
    """
    this class is to fix issue with original ModelCheckpoint where saving model every N epochs may be a bad idea
    instead of save best after at least N epochs
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            # self.epochs_since_last_save = 0  # we want save next best model, not best if every
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.epochs_since_last_save = 0  # modification
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 1:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                self.epochs_since_last_save = 0  # modification
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        super().on_train_begin(logs)


class WeightsWriter(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.file_path, overwrite=True)


class VerboseEarlyStopping(EarlyStopping):
    def __init__(self, **kwargs):
        self.best = 0  # to suppress IDE warnings
        self.best_epoch_nr = -1
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            # my verbosity addon
            print(" {:<20} {:<20} {:<20} {:<5} {:<5}".
                  format(*'previous_best current_best change wait_epochs patience'.split()))
            return

        if self.monitor_op(current - self.min_delta, self.best):
            # my verbosity addon
            self.best_epoch_nr = epoch
            if self.verbose > 1:
                if self.best - current > abs(self.min_delta):
                    print(" {:<20} {:<20} {:<20} {:<5} {:<5}".
                          format(self.best, current, self.best-current, self.wait, self.patience))
            elif self.verbose > 0:
                if self.best - current > abs(self.min_delta)*1.4:
                    print(" {:<20} {:<20} {:<20} {:<5} {:<5}".
                          format(self.best, current, self.best-current, self.wait, self.patience))
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                        print("best epoch {} best score {}".format(self.best_epoch_nr, self.best))
                    self.model.set_weights(self.best_weights)


class TensorBoardTemplate(TensorBoard):
    def __init__(self, log_dir_template, log_lr=True, **kwargs):
        self.log_lr = log_lr
        self.log_dir_template = log_dir_template
        super().__init__(**kwargs)

    def update_log_dir(self, *args, **kwargs):
        self.log_dir = self.log_dir_template.format(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if self.log_lr:
            logs = logs or {}
            logs['lr'] = K.get_value(self.model.optimizer.lr)
        super(TensorBoardTemplate, self).on_epoch_end(epoch, logs)


class TensorBoardCountRuns(TensorBoard):
    def __init__(self, log_dir_template, **kwargs):
        super().__init__(**kwargs)
        self.runs_count = 0
        self.log_dir_template = log_dir_template
        self.update_log_dir()

    def update_log_dir(self):
        self.runs_count += 1
        self.log_dir = self.log_dir_template.format(run=self.runs_count)

