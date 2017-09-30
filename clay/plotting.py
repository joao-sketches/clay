import numpy as np
from matplotlib import pyplot as plt


def plot_learning_curve(train_sizes, train_scores, test_scores, **kwargs):
    """
    Plot the learning curve calculated by a function previous used, e.g. `sklearn.model_selection.learning_curve`.

    For styling configuration see :code:`kwargs` accepted parameters, also values set to pyplot context prior to the call
    will affect this function

    :param train_sizes: the train size value returned from :code:`sklearn.model_selection.learning_curve`
    :type train_sizes: array_like
    :param train_scores: the train score value returned from :code:`sklearn.model_selection.learning_curve`
    :type train_scores: array_like
    :param test_scores: the test score value returned from :code:`sklearn.model_selection.learning_curve`
    :type test_scores: array_like
    :keyword alpha: transparency to plot line, defaults to: :code:`0.1`
    :keyword train_color: color for train score plot line, defaults to: "green"
    :keyword test_color: color for test score plot line, defaults to: "blue"
    :keyword train_label: label for train set, defaults to: "Training Score"
    :keyword test_label: label for test set, defaults to: "Cross-validation Score"


    :return: Returns the plot context
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    alpha = kwargs.pop('alpha', 0.1)
    train_color = kwargs.pop('train_color', 'g')
    test_color = kwargs.pop('test_color', 'b')
    train_label = kwargs.pop('train_label', 'Training Score')
    test_label = kwargs.pop('train_label', 'Cross-validation Score')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, color=train_color, alpha=alpha)

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, color=test_color, alpha=alpha)

    plt.plot(train_sizes, train_scores_mean, 'o-', color=train_color,
             label=train_label)

    plt.plot(train_sizes, test_scores_mean, 'o-', color=test_color,
             label=test_label)

    return plt


def plot_validation_curve(train_scores, test_scores, param_range, **kwargs):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    alpha = kwargs.pop('alpha', 0.2)
    train_color = kwargs.pop('train_color', 'g')
    test_color = kwargs.pop('test_color', 'b')
    line_width = kwargs.pop('line_width', kwargs.pop('lw', 2))
    train_label = kwargs.pop('train_label', 'Training Score')
    test_label = kwargs.pop('test_label', 'Cross-validation Score')

    plt.semilogx(param_range, train_scores_mean, label=train_label,
                 color=train_color, lw=line_width)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=alpha,
                     color=train_color, lw=line_width)

    plt.semilogx(param_range, test_scores_mean, label=test_label,
                 color=test_color, lw=line_width)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=alpha,
                     color=test_color, lw=line_width)

    return plt
