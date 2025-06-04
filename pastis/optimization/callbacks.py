import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import os
from .utils_poisson import find_beads_to_remove
from .multiscale_optimization import decrease_lengths_res, decrease_struct_res


class Callback(object):
    """An object that adds functionality during optimization.

    A callback is a function or group of functions that can be executed during
    optimization. A callback can be called at three stages-- the
    beginning of optimization, at the end of each epoch (or iteration), and at
    the end of optimization. Users can define any functions that they wish in
    the corresponding functions.Compute objective constraints.

    Parameters
    ----------
    lengths : array_like of int
        Number of beads per homolog of each chromosome at full resolution.
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    counts : list of CountsMatrix subclass instances
        Preprocessed counts data.
    multiscale_factor : int, optional
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    history : dict of list, optional
        Previously generated history logs, to be added to during this
        optimization.
    analysis_function : function, optional
        Analysis function to be executed before history is logged.
        The function should only take `self` as an input, and should return a
        dictionary of items that will be added to the history logs.
    frequency : int or dict, optional
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "history" (logs history), "print"
        (prints time and current objective value), "save" (saves current
        structure to file).
    on_training_begin : function, optional
        Function to be executed at the beginning of optimization.
    on_training_end : function, optional
        Function to be executed at the end of optimization.
    on_epoch_end : function, optional
        Function to be executed at the end of each epoch.
    directory : str, optional
        Directory in which to save structures generated during optimization.
    struct_true : array of float, optional
        True structure, to be used by `analysis_function`.
    alpha_true : array of float, optional
        True alpha, to be used by `analysis_function`.
    verbose : bool, optional
        Verbosity.

    Attributes
    ----------
    lengths : array of int
        Number of beads per homolog of each chromosome at full the current
        resolution (indicated by `multiscale_factor`).
    ploidy : {1, 2}
        Ploidy, 1 indicates haploid, 2 indicates diploid.
    multiscale_factor : int
        Factor by which to reduce the resolution. A value of 2 halves the
        resolution. A value of 1 indicates full resolution.
    torm : array of bool
        Beads that should be removed (set to NaN) in the structure.
    frequency : dict
        Frequency of iterations at which operations are performed. Each key
        indicates an operation, options: "history" (logs history), "print"
        (prints time and current objective value), "save" (saves current
        structure to file).
    analysis_function : function or None
        Analysis function to be executed before history is logged.
        The function should only take `self` as an input, and should return a
        dictionary of items that will be added to the history logs.
    on_training_begin_ : function or None
        Function to be executed at the beginning of optimization.
    on_training_end_ : function or None
        Function to be executed at the end of optimization.
    on_epoch_end_ : function or None
        Function to be executed at the end of each epoch.
    directory : str
        Directory in which to save structures generated during optimization.
    struct_true : array of float or None
        True structure, to be used by `analysis_function`.
    alpha_true : array of float or None
        True alpha, to be used by `analysis_function`.
    verbose : bool
        Verbosity.
    history : dict of list
        History logs generated during optimization. By default, history includes
        the following information and keys, respectively: iteration ("iter"),
        alpha ("alpha"), multiscale_factor ("multiscale_factor"),
        current iteration of alpha/structure optimization ("alpha_loop"),
        relative orientation of each chromosome ("opt_type").
    opt_type : {"structure", "alpha", "chrom_reorient", None}
        Type of optimization being performed. Options: 3D chromatin structure
        ("structure"), alpha ("alpha"), relative orientation of each chromosome
        ("chrom_reorient").
    alpha_loop : int or None
        Current iteration of alpha/structure optimization.
    epoch : int
        Current epoch (iteration) of optimization.
    time : str
        Time since optimization began.
    structures : list of array of float or None
        Current 3D chromatin structure(s).
    alpha : float or None
        Current biophysical parameter of the transfer function used in
        converting counts to wish distances.
    Xi : float or array of float or None
        Current values being optimized (structure, alpha, or chromosome
        orientation).
    orientation : array of float or None
        Current values for the relative orientation of each chromosome
        (for `opt_type` == "chrom_reorient").
    time_start : timeit.default_timer instance
        Time at which optimization began.

    """

    def __init__(self, lengths, ploidy, counts=None, multiscale_factor=1,
                 history=None, analysis_function=None, frequency=None,
                 on_training_begin=None, on_training_end=None,
                 on_epoch_end=None, directory=None, struct_true=None,
                 alpha_true=None, verbose=False):
        self.ploidy = ploidy
        self.multiscale_factor = multiscale_factor
        self.lengths = decrease_lengths_res(lengths, multiscale_factor)
        if counts is None:
            self.torm = np.full((lengths.sum() * self.ploidy), False)
        else:
            self.torm = find_beads_to_remove(
                counts, self.lengths.sum() * self.ploidy)
        self.analysis_function = analysis_function
        if frequency is None or isinstance(frequency, int):
            self.frequency = {'print': frequency,
                              'history': frequency, 'save': frequency}
        else:
            if not isinstance(frequency, dict) or any(
                    [k not in ('print', 'history', 'save') for k in frequency.keys()]):
                raise ValueError("Callback frequency must be None, int, or dict"
                                 " with keys = (print, history, save).")
            self.frequency = {'print': None, 'history': None, 'save': None}
            for k, v in frequency.items():
                self.frequency[k] = v
        self.on_training_begin_ = on_training_begin
        self.on_training_end_ = on_training_end
        self.on_epoch_end_ = on_epoch_end
        if directory is None:
            directory = ''
        self.directory = directory
        if struct_true is not None:
            self.struct_true = decrease_struct_res(
                struct_true, multiscale_factor=multiscale_factor,
                lengths=lengths)
        else:
            self.struct_true = None
        self.alpha_true = alpha_true
        self.verbose = verbose

        if history is None:
            self.history = {}
        elif isinstance(history, dict) and all(
                [isinstance(v, list) for v in history.values()]):
            self.history = history
        else:
            raise ValueError("History must be dictionary of lists")

        self.opt_type = None
        self.alpha_loop = None
        self.epoch = -1
        self.time = '0:00:00.0'
        self.structures = None
        self.alpha = None
        self.X = None
        self.orientation = None
        self.time_start = None

    def _set_structures(self, structures):
        self.structures = [struct.copy().reshape(-1, 3)
                           for struct in structures]
        for i in range(len(structures)):
            self.structures[i][self.torm] = np.nan

    def _check_frequency(self, frequency, last_epoch=False):
        output = False
        if frequency is not None and frequency:
            if not last_epoch and self.epoch % frequency == 0:
                output = True
            elif last_epoch and self.epoch % frequency != 0:
                output = True
        return output

    def _print(self, last_epoch=False):
        """Prints loss every given number of epochs."""

        if self._check_frequency(self.frequency['print'], last_epoch):
            info_dict = {'At iterate': ' ' * (6 - len(str(self.epoch))) + str(
                self.epoch), 'f= ': '%.6g' % self.obj['obj'],
                'time= ': self.time}
            print('\t\t'.join(['%s%s' % (k, v)
                               for k, v in info_dict.items()]), flush=True)
            if self.epoch == 10:
                print('. . .', flush=True)
            print('', flush=True)

    def _save_X(self):
        """This will save the model to disk every given number of epochs."""

        if self._check_frequency(self.frequency['save']):
            X_list = self.X
            if not isinstance(X_list, list):
                X_list = [X_list]
            for i in range(len(X_list)):
                if len(X_list) == 1:
                    filename = os.path.join(
                        self.directory, 'inferred_%s.epoch_%07d.txt'
                                        % (self.opt_type, self.epoch))
                else:
                    filename = os.path.join(
                        self.directory, 'struct%d.inferred_%s.epoch_%07d.txt'
                                        % (i, self.opt_type, self.epoch))
                if self.verbose:
                    print("[%d] Saving model checkpoint to %s" %
                          (self.epoch, filename))
                np.savetxt(filename, X_list[i])

    def _log_history(self, last_epoch=False):
        """Keeps a history of the loss and other values every given number of epochs."""

        if self._check_frequency(self.frequency['history'], last_epoch):
            if not isinstance(self.alpha, np.ndarray) or self.alpha.shape == () or self.alpha.shape[0] == 1:
                alpha = float(self.alpha)
            else:
                alpha = ','.join(map(str, self.alpha))
            to_log = [('iter', self.epoch), ('alpha', alpha),
                      ('alpha_loop', self.alpha_loop),
                      ('opt_type', self.opt_type),
                      ('multiscale_factor', self.multiscale_factor),
                      ('seconds', self.seconds)]
            to_log.extend(list(self.obj.items()))

            if self.analysis_function is not None:
                to_log.extend(self.analysis_function(self).items())

            for k, v in to_log:
                if k in self.history:
                    self.history[k].append(v)
                else:
                    self.history[k] = [v]

    def on_training_begin(self, opt_type=None, alpha_loop=None):
        """Functionality to add to the beginning of optimization.

        This method will be called at the beginning of the optimization
        procedure.

        Parameters
        ----------
        opt_type : {"structure", "alpha", "chrom_reorient", None}
            Type of optimization being performed. Options: 3D chromatin
            structure ("structure"), alpha ("alpha"), relative orientation of
            each chromosome ("chrom_reorient").
        alpha_loop : int
            Current iteration of alpha/structure optimization.
        """

        if opt_type is None:
            self.opt_type = 'structure'
        else:
            self.opt_type = opt_type
        self.alpha_loop = alpha_loop
        self.epoch = -1
        self.seconds = 0
        self.time = '0:00:00.0'
        self.structures = None
        self.alpha = None
        self.orientation = None
        if self.on_training_begin_ is not None:
            self.on_training_begin_(self)
        self.time_start = timer()

    def on_epoch_end(self, obj_logs, structures, alpha, Xi):
        """Functionality to add to the end of each epoch.

        This method will be called at the end of each epoch during the
        iterative optimization procedure.

        Parameters
        ----------
        obj_logs : dict of float
            Current objective function. Each component of the objective is
            indicated by a separate dictionary item.
        structures : list of array of float
            Current 3D chromatin structure(s).
        alpha : float
            Current biophysical parameter of the transfer function used in
            converting counts to wish distances.
        Xi : float or array of float
            Current values being optimized (structure, alpha, or chromosome
            orientation).
        """

        self.epoch += 1
        self.seconds = round(timer() - self.time_start, 1)
        current_time = str(timedelta(seconds=self.seconds)).split('.')
        if len(current_time) == 1:
            self.time = current_time[0] + '.0'
        else:
            self.time = current_time[
                0] + str(float('0.' + current_time[1])).lstrip('0')
        self.obj = obj_logs

        self.X = Xi
        if self.opt_type != 'alpha' or self.epoch == 0:
            self._set_structures(structures)
        self.alpha = alpha
        if self.opt_type == 'chrom_reorient':
            self.orientation = Xi

        self._print()
        self._log_history()
        self._save_X()
        if self.on_epoch_end_ is not None:
            self.on_epoch_end_(self)

    def on_training_end(self):
        """Functionality to add to the end of optimization.

        This method will be called at the end of the optimization procedure.
        """

        self._print(last_epoch=True)
        self._log_history(last_epoch=True)
        if self.on_training_end_ is not None:
            self.on_training_end_(self)
