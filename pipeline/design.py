#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:00:05 2018

@author: goncalves1
"""
import os
import types
import shutil
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

matplotlib.rcParams.update({'font.size': 11})
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams.update({'figure.autolayout': True})

from abc import ABCMeta, abstractmethod
from UTILS import config, performance_metrics, utils
from UTILS.Logger import Logger
from hp_optimization import optimize_hyper_params
import numpy as np


class Dataset(object):
    """ """
    __metaclass__ = ABCMeta

    def __init__(self, dataset_name):
        """."""
        self.name = None
        self.data = None
        self.dataset_name = dataset_name

    @abstractmethod
    def prepare_data(self):
        """."""
        pass

    @abstractmethod
    def shuffle_and_split(self):
        """."""
        pass


class ModelTraining(object):
    """ Train all models on the dataset provided. Methods could be
    either of STL or MTL types. For STL, the same method will be
    applied for every dataset into the MTL-Dataset. For the MTL methods,
    it will be applied once for all datasets jointly. Once trained,
    the models will be tested on a hold-out set and ther performances
    are computed and stored for later analysis.
    """

    def __init__(self, name):

        assert isinstance(name, str)

        self.name = name
        self.dataset = None
        self.methods = None
        self.metrics = None
        self.nb_runs = -1

        self.logger = Logger()

    def execute(self, dataset, methods, metrics, nruns=1):  # , report_only=False):

        self.__check_inputs(dataset, methods, metrics, nruns)
        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics

        self.nb_runs = nruns

        # set experiment output directory
        directory = os.path.join(config.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)
        # make a new directory with experiment name
        os.makedirs(directory)

        # experiment log file will be save in 'directory'
        self.logger.set_path(directory)
        self.logger.setup_logger('{}.log'.format(self.name))
        self.logger.info('Experiment directory created.')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        results_runs = dict()
        for r_i in range(self.nb_runs):

            self.logger.info('Executing \'Run {}\'.'.format(r_i + 1))

            # shuffle and re-split the data between training and test
            self.dataset.shuffle_and_split()

            run_directory = os.path.join(directory, 'run_{}'.format(r_i + 1))

            results_runs['run_{}'.format(r_i + 1)] = dict()
            # execute all methods passed through 'methods' attribute
            for method in self.methods:

                self.logger.info('Running {}.'.format(method.name))

                # set method's output directory
                method_directory = os.path.join(run_directory, method.__str__())
                # create directory to save method's results/logs
                os.makedirs(method_directory)

                # inform output directory path to the method
                method.set_output_directory(method_directory)

                # check model type: feature_based or non_feature_based
                if method.type == 'feature_based':
                    if method.paradigm == 'stl':
                        result_method = {}
                        for k in self.dataset.datasets:
                            method.fit(self.dataset.data['train']['x'][k],
                                       y=self.dataset.data['train']['y'][k],
                                       cat_point=self.dataset.cat_point)
                            y_pred = method.predict(self.dataset.data['test']['x'][k])
                            y_true = self.dataset.testRatings[k]
                            if method.output_shape == 'array':
                                y_pred = utils.predMatrix(y_true, y_pred)  # move back to rating matrix
                            # dict to save performance metrics for the t-th task
                            result_method[k] = {}
                            for met in self.metrics:
                                result_method[k][met] = metric_func[met](y_pred, y_true)

                    elif method.paradigm == 'mtl':
                        method.fit(self.dataset.data['train']['x'],
                                   self.dataset.data['train']['y'],
                                   cat_point=self.dataset.cat_point)

                       #y_pred = method.predict(self.dataset.data['train']['x']) # bug
                        y_pred = method.predict(self.dataset.data['test']['x']) 
                        result_method = {}
                        for k in self.dataset.datasets:
                            y_true_k = self.dataset.testRatings[k]
                            y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to rating matrix
                            result_method[k] = {}
                            for met in self.metrics:
                                result_method[k][met] = metric_func[met](y_pred_k, y_true_k)

                    else:
                        raise ValueError('Unknown paradigm: {}'.format(method.paradigm))

                elif method.type == 'non_feature_based':

                    if method.paradigm == 'stl':
                        result_method = {}                            
                        for k in self.dataset.datasets:
                            method.fit(self.dataset.trainRatings[k])
                            y_pred_k = method.predict(self.dataset.testRatings[k])
                            y_true_k = self.dataset.testRatings[k]
                            if method.output_shape == 'array':
                                y_pred_k = utils.predMatrix(y_true_k, y_pred_k)  # move back to rating matrix
                            # store results to dict of all performances
                            result_method[k] = {}

                            for met in self.metrics:
                                result_method[k][met] = metric_func[met](y_pred_k, y_true_k)

                    elif method.paradigm == 'mtl':
                        method.fit(self.dataset.trainRatings)
                        y_pred = method.predict(self.dataset.testRatings)
                        result_method = {}
                        for k in self.dataset.datasets:
                            y_true_k = self.dataset.testRatings[k]
                            if method.output_shape == 'array':
                                y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to rating matrix
                            result_method[k] = {}
                            for met in self.metrics:
                                result_method[k][met] = metric_func[met](y_pred[k], y_true_k)

                    else:
                        raise ValueError('Unknown paradigm: {}'.format(method.paradigm))
                else:
                    raise ValueError('Unknown type %s' % (method.type))

                results_runs['run_{}'.format(r_i + 1)][method.__str__()] = result_method
        with open(os.path.join(directory, 'performance.json'), 'w') as fp:
            json.dump(results_runs, fp, indent=4, sort_keys=True)

    def generate_report(self):
        # read results from experiment folder and store it into a dataframe
        df = self.__read_experiment_results()
        print(df.groupby(['Dataset', 'Method', 'Metric']).agg(['mean', 'std']))

        # set output pdf name
        pdf_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_report.pdf'.format(self.name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        # call several plot functions
        self._performance_boxplots(df, pdf)

        # close pdf file
        pdf.close()
        
    def getResultsWrapper(self):
        """
        wrapper for testing cases so we can check accuracy, without messing up private method
        and breaking abstraction
        """
        df = self.__read_experiment_results()
        return df

    def generate_method_report(self):
        # read results from experiment folder and store it into a dataframe
        df = self.__read_experiment_results()
        line_df, drug_df = self.__get_list_metrics()

        # set output pdf name
        pdf_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_report_methods.pdf'.format(self.name))

        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)

        # call grid plot function
        self._performance_grid(df, pdf)
        self._metric_by_drug_bar(drug_df, pdf)
        self._metric_by_line_scatter(line_df)
        self.sim_maps(self.methods,pdf)


        # close pdf file
        pdf.close()

    def __check_inputs(self, dataset, methods, metrics, nb_runs):
        # make sure all inputs have expected values and types
        assert isinstance(dataset, Dataset)

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)
        assert len(metrics) > 0

        # check if all methods are valid (instance of Method class)
        # for method in methods:
        # assert isinstance(method, BaseEstimator)

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert nb_runs > 0

    def __get_list_metrics(self):
        """ returns DFs for list shaped metrics like rmse_by_cell_line """
        experiment_dir = os.path.join(config.path_to_output, self.name)
        with open(os.path.join(experiment_dir, 'performance.json'), 'r') as fh:
            data = json.load(fh)

        perf_list = list()
        perf_list2 = list()
        for run in data.keys():
            for method in data[run].keys():
                for db in data[run][method].keys():
                    for metric in data[run][method][db].keys():
                        if "line" in metric:
                            value = data[run][method][db][metric]
                            perf_list2.append([run, method, db, metric, value])
                        elif "drug" in metric:
                            value = data[run][method][db][metric]
                            perf_list.append([run, method, db, metric, value])
        column_names = ['Run', 'Method', 'Dataset', 'Metric', 'Value']
        df_lines = pd.DataFrame(perf_list2, columns=column_names)
        df_drugs = pd.DataFrame(perf_list, columns=column_names)
        return df_lines, df_drugs

    def __read_experiment_results(self):
        """ Read in performance results from the json file.
        
        Returns:
            pandas.DataFrame: runs, methods, metrics, and perf in a DF
        """
        experiment_dir = os.path.join(config.path_to_output, self.name)
        with open(os.path.join(experiment_dir, 'performance.json'), 'r') as fh:
            data = json.load(fh)

        perf_list = list()
        for run in data.keys():
            for method in data[run].keys():
                for db in data[run][method].keys():
                    for metric in data[run][method][db].keys():
                        if "line" not in metric and "drug" not in metric:
                            value = data[run][method][db][metric]
                            perf_list.append([run, method, db, metric, value])
        column_names = ['Run', 'Method', 'Dataset', 'Metric', 'Value']
        df = pd.DataFrame(perf_list, columns=column_names)
        return df

    def _performance_boxplots(self, df, pdf):
        """ Create boxplot with performance of all methods per metric.

        Args:
            df (pandas.DataFrame): Dataframe with runs/models/performance
            pdf (obj): pdf object to save the plot on
        """
        for metric in self.metrics:
            if "line" not in metric and "drug" not in metric:
                df_a = df[df['Metric'] == metric].copy()
                g = sns.boxplot(x="Method", y="Value", data=df_a)
                plt.title("Root Mean Squared Prediction Error")
                plt.ylabel("Error")
                plt.tight_layout()
                pdf.savefig(g.figure)
                plt.clf()

    def sim_maps(self, methods,pdf):
        for predictor in methods:
            if "KNN" in predictor.name:
                sim_matrix = np.array(predictor.similarity_matrix())
                f, ax = plt.subplots(figsize=(9, 6))
                g = sns.heatmap(sim_matrix, linewidths=.5, ax=ax)
                plt.title(predictor.name + " similarity heatmap for last dataset")
                plt.xlabel("Cell lines")
                plt.ylabel("Cell lines")
                plt.tight_layout()
                pdf.savefig(g.figure)
                plt.clf()


    def _metric_by_drug_bar(self, df, pdf):
        """Function to create a bar plot for every metric showing that metric across all drugs
        and all methods in the experiment. There is a definitely a nicer way to code this but pandas
        groupby doesn't work so great with lists, would need a pivot I think.
        TODO: get confidence bars with stddev
        Parameters
        ----------------------------------------
        df: pandas dataframe with all data from every method, run, and metric
        pdf: a pdf to save figures to
        Outputs
        ----------------------------------------
        none, this just saves the figures to the pdf
        """
        for metric in np.unique(df['Metric']):     # only one metric right now but keep this here so we can generalize
            y_vals = np.array([])
            x_vals = np.array([])
            hues = np.array([])
            for method in np.unique(df['Method']):
                curr_df = df[df['Metric'] == metric]
                curr_df = curr_df[curr_df['Method'] == method]
                scores = curr_df['Value'].values.tolist()
                scores_by_runs = np.mean(np.array(scores), axis=0)
                y_vals = np.append(y_vals, scores_by_runs)
                x_vals = np.append(x_vals, np.arange(10))
                hues = np.append(hues, (np.repeat(method,len(scores_by_runs))))
            data = {'yvalues': y_vals, 'xvalues': x_vals, 'hue': hues}
            plot_df = pd.DataFrame(data)
            f, ax = plt.subplots(figsize=(9, 6))
            g = sns.barplot(x="xvalues", y="yvalues", hue="hue", data=plot_df, palette="muted", ax=ax)
            plt.title("Total " + metric + " across each drug line")
            plt.xlabel('Drug #'), plt.ylabel(metric), plt.tight_layout()
            pdf.savefig(g.figure)
            plt.clf()

    def _metric_by_line_scatter(self, df):
        """Function to create a bar plot for every metric showing that metric across all drugs
        and all methods in the experiment. There is a definitely a nicer way to code this but pandas
        groupby doesn't work so great with lists, would need a pivot I think.
        Parameters
        ----------------------------------------
        df: pandas dataframe with all data from every method, run, and metric
        pdf: a pdf to save figures to
        Outputs
        ----------------------------------------
        none, this just saves the figures to the pdf
        """
        y_vals = np.array([])
        x_vals = np.array([])
        dset = np.array([])
        hues = np.array([])
        dsets = np.unique(df['Dataset'])
        for i in range(len(dsets)):
            # only one metric right now but keep this here so we can generalize
            for method in np.unique(df['Method']):
                curr_df = df[df['Method'] == method]
                curr_df = curr_df[curr_df['Dataset'] == dsets[i]]
                scores = np.array(curr_df['Value'].values.tolist())
                scores_by_runs = np.mean(scores, axis=0)
                #print("SCORES by runs shape", scores_by_runs.shape)
                y_vals = np.append(y_vals, scores_by_runs)
                x_vals = np.append(x_vals, np.arange(len(scores_by_runs)))
                hues = np.append(hues, (np.repeat(method, len(scores_by_runs))))
                dset = np.append(dset, np.repeat(dsets[i], len(scores_by_runs)) )
        data = {'yvalues': y_vals, 'xvalues': x_vals, 'hue': hues, 'dataset': dset}
        plot_df = pd.DataFrame(data)
        g = sns.FacetGrid(plot_df, col="dataset", hue="hue", sharex=False, sharey=False)
        g.map(plt.plot, "xvalues", "yvalues", alpha=.7)
        plt.xlabel("cell line #"), plt.ylabel("rmse")
        plt.tight_layout()
        g.add_legend();
        g.savefig('../outputs/experiment_001x/rmse_by_line.png')

    def _performance_grid(self, df, pdf):
        df = df[df['Metric'] == 'rmse']
        df_grouped = df.groupby(['Dataset', 'Method']).agg(['mean'])
        df_grouped.columns = df_grouped.columns.droplevel(0)
        df_grouped = df_grouped.reset_index()
        df_piv = df_grouped.pivot(index='Method', columns='Dataset', values='mean')
        ylabels = np.unique(df_piv.index.to_numpy())
        xlabels = np.unique(df_piv.columns.to_numpy())
        data = df_piv.to_numpy()

        f, ax = plt.subplots(figsize=(9, 6))
        g = sns.heatmap(data, annot=True, fmt="f", cmap='Reds', linewidths=.5, ax=ax )
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
        plt.title("RMSE by Dataset and Method for STL")
        plt.xlabel('Dataset')
        plt.ylabel('Method used')
        plt.tight_layout()
        pdf.savefig(g.figure)
        plt.clf()

class VaryingMissingData(object):
    """ Train and test models on a set of missing value
    percentages. As a result, a curver of performance for 
    each method is generated.
    """

    def __init__(self, name):

        assert isinstance(name, str)

        self.name = name
        self.dataset = None
        self.methods = None
        self.metrics = None
        self.nb_runs = -1

        self.logger = Logger()

    def execute(self, dataset, methods, metrics, missing_perc,
                nruns=1, report_only=False):

        self.__check_inputs(dataset, methods, metrics, nruns)
        self.dataset = dataset
        self.methods = methods
        self.metrics = metrics

        if report_only:
            return

        self.nb_runs = nruns

        # set experiment output directory
        directory = os.path.join(config.path_to_output, self.name)
        # if directory already exists, then delete it
        if os.path.exists(directory):
            shutil.rmtree(directory)
        # make a new directory with experiment name
        os.makedirs(directory)

        # experiment log file will be save in 'directory'
        self.logger.set_path(directory)
        self.logger.setup_logger('{}.log'.format(self.name))
        self.logger.info('Experiment directory created.')

        # get list of available metrics
        metric_func = {a: performance_metrics.__dict__.get(a)
                       for a in dir(performance_metrics)
                       if isinstance(performance_metrics.__dict__.get(a),
                                     types.FunctionType)}

        results_missing = dict()
        for mi in missing_perc:

            str_mi = str(round(mi, 2))
            # make a new directory with experiment name
            mi_directory = os.path.join(directory, 'missing_%s' % str_mi)
            os.makedirs(mi_directory)

            results_missing[str_mi] = dict()
            self.dataset.set_test_split(mi)

            for r_i in range(self.nb_runs):

                self.logger.info('Executing \'Run {}\'.'.format(r_i + 1))

                # shuffle and re-split the data between training and test
                self.dataset.shuffle_and_split()
                # directory to store a particular run of all methods
                run_directory = os.path.join(mi_directory, 'run_{}'.format(r_i + 1))
                # store all runs for this particular missing data percentage
                results_missing[str_mi]['run_{}'.format(r_i + 1)] = dict()

                # execute all methods passed through 'methods' attribute
                for method in self.methods:

                    self.logger.info('Running {}.'.format(method.name))

                    # set method's output directory
                    method_directory = os.path.join(run_directory, method.__str__())
                    # create directory to save method's results/logs
                    os.makedirs(method_directory)

                    # inform output directory path to the method
                    method.set_output_directory(method_directory)

                    # check model type: feature_based or non_feature_based
                    if method.type == 'feature_based':
                        if method.paradigm == 'stl':
                            result_method = {}
                            for k in self.dataset.datasets:
                                method.fit(self.dataset.data['train']['x'][k],
                                           y=self.dataset.data['train']['y'][k])
                                y_pred = method.predict(self.dataset.data['test']['x'][k])
                                y_true = self.dataset.testRatings[k]
                                if method.output_shape == 'array':
                                    y_pred = utils.predMatrix(y_true, y_pred)  # move back to rating matrix
                                # dict to save performance metrics for the t-th task
                                result_method[k] = {}
                                for met in self.metrics:
                                    result_method[k][met] = metric_func[met](y_pred, y_true)

                        elif method.paradigm == 'mtl':
                            method.fit(self.dataset.data['train']['x'],
                                       self.dataset.data['train']['y'])

                            y_pred = method.predict(self.dataset.data['train']['x'])
                            result_method = {}
                            for k in self.dataset.datasets:
                                y_true_k = self.dataset.testRatings[k]
                                y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to rating matrix
                                result_method[k] = {}
                                for met in self.metrics:
                                    result_method[k][met] = metric_func[met](y_pred_k, y_true_k)

                        else:
                            raise ValueError('Unknown paradigm: {}'.format(method.paradigm))

                    elif method.type == 'non_feature_based':

                        if method.paradigm == 'stl':
                            result_method = {}
                            for k in self.dataset.datasets:
                                method.fit(self.dataset.trainRatings[k])
                                y_pred_k = method.predict(self.dataset.testRatings[k])
                                y_true_k = self.dataset.testRatings[k]
                                if method.output_shape == 'array':
                                    y_pred_k = utils.predMatrix(y_true_k, y_pred_k)  # move back to rating matrix
                                # store results to dict of all performances
                                result_method[k] = {}
                                for met in self.metrics:
                                    result_method[k][met] = metric_func[met](y_pred_k, y_true_k)

                        elif method.paradigm == 'mtl':
                            method.fit(self.dataset.trainRatings)
                            y_pred = method.predict(self.dataset.testRatings)
                            result_method = {}
                            for k in self.dataset.datasets:
                                y_true_k = self.dataset.testRatings[k]
                                if method.output_shape == 'array':
                                    y_pred_k = utils.predMatrix(y_true_k, y_pred[k])  # move back to rating matrix
                                result_method[k] = {}
                                for met in self.metrics:
                                    result_method[k][met] = metric_func[met](y_pred[k], y_true_k)

                        else:
                            raise ValueError('Unknown paradigm: {}'.format(method.paradigm))
                    else:
                        raise ValueError('Unknown type %s' % (method.type))

                    results_missing[str_mi]['run_{}'.format(r_i + 1)][method.__str__()] = result_method

        with open(os.path.join(directory, 'performance.json'), 'w') as fp:
            json.dump(results_missing, fp, indent=4, sort_keys=True)

    def generate_report(self):
        # read results from experiment folder and store it into a dataframe
        df = self.__read_experiment_results()
        print(df.groupby(['Missing', 'Method', 'Metric']).agg(['mean', 'std']))
        # set output pdf name
        pdf_filename = os.path.join(config.path_to_output,
                                    self.name,
                                    '{}_report.pdf'.format(self.name))
        # create a pdf object to place plots
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        # call several plot functions
        self._performance_boxplots(df, pdf)
        # close pdf file
        pdf.close()

    def __check_inputs(self, dataset, methods, metrics, nb_runs):
        # make sure all inputs have expected values and types
        assert isinstance(dataset, Dataset)

        # make sure it received a list of methods
        if not isinstance(methods, list):
            methods = list(methods)
        assert len(methods) > 0

        # make sure it received a list of metrics
        if not isinstance(metrics, list):
            metrics = list(metrics)
        assert len(metrics) > 0

        # check if all methods are valid (instance of Method class)
        # for method in methods:
        # assert isinstance(method, BaseEstimator)

        # get existing list of available performance metrics
        existing_metrics = [a for a in dir(performance_metrics)
                            if isinstance(performance_metrics.__dict__.get(a),
                                          types.FunctionType)]
        # check if all metrics are valid (exist in performance_metrics module)
        for metric in metrics:
            assert metric in existing_metrics

        # number of runs has to be larger then 0
        assert nb_runs > 0

    def __read_experiment_results(self):
        """ Read in performance results from the json file.

        Returns:
            pandas.DataFrame: runs, methods, metrics, and perf in a DF
        """
        experiment_dir = os.path.join(config.path_to_output, self.name)
        with open(os.path.join(experiment_dir, 'performance.json'), 'r') as fh:
            data = json.load(fh)

        perf_list = list()
        for mi in data.keys():
            for run in data[mi].keys():
                for method in data[mi][run].keys():
                    for dataset in data[mi][run][method].keys():
                        for metric in data[mi][run][method][dataset].keys():
                            value = data[mi][run][method][dataset][metric]
                            perf_list.append([mi, dataset, run, method, metric, value])

        column_names = ['Missing', 'Dataset', 'Run', 'Method', 'Metric', 'Value']
        df = pd.DataFrame(perf_list, columns=column_names)
        return df

    def _performance_boxplots(self, df, pdf):
        """ Create boxplot with performance of all methods per metric.
        
        Args:
            df (pandas.DataFrame): Dataframe with runs/models/performance
            pdf (obj): pdf object to save the plot on
        """
        for metric in self.metrics:
            num_datasets = len(df['Dataset'].unique())
            fig, ax = plt.subplots(1, num_datasets, squeeze=False, sharey=True)
            for i, dataset in enumerate(df['Dataset'].unique()):
                df_a = df[(df['Metric'] == metric) & (df['Dataset'] == dataset)].copy()
                sns.lineplot(x='Missing', y='Value', hue='Method', data=df_a, ax=ax[0, i])
                ax[0, i].set_title(dataset)
                ax[0, i].set_ylabel(metric)
                ax[0, i].xaxis.set_major_locator(plt.MaxNLocator(4))

                # plt.locator_params(nbins=4)
                # plt.xticks(rotation=70)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.clf()
