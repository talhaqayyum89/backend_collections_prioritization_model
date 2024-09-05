def get_confusion_matrix(select_best):
    confusion_matrix = plot_model(select_best, 'confusion_matrix')

    return confusion_matrix


def get_roc_auc(select_best):
    roc_auc = plot_model(select_best, 'auc')

    return roc_auc


def get_class_report(select_best):
    class_report = plot_model(select_best, 'class_report')

    return class_report



