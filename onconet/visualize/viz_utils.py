import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import sklearn.metrics


SAVE_PLOT_STR="Saved figure for {} at {}"
ERR_PLOT_STR="ERROR! Failed to save figure for {} at {}. Exception {}}"

def plot_losses(args):
    epochs = range(args.epochs)
    losses = args.epoch_details

    for key in losses:
        save_path = "{}.{}.png".format(args.results_dir, key)
        loss = losses[key]
        try:
            plt.plot(epochs, loss)
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.savefig(save_path)
            plt.close()
            print(SAVE_PLOT_STR.format(key, save_path))
        except Exception, e:
            print(ERR_PLOT_STR.format(
                key, save_path, e))

def plot_pred_stats(golds_percent, pred_percent, save_path):
    import matplotlib
    matplotlib.use('Agg') 
    from matplotlib import pyplot as plt
    classes = ['Fatty', 'Scattered', 'Hetrogenous', 'Dense']
    width = 0.35
    ind = np.arange(len(pred_percent))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rec1 = ax.bar(ind-width/2, pred_percent.values(), width, align='center', label='Model')
    rec2 = ax.bar(ind+width/2, golds_percent.values(), width, align='center', label='Human')
    ax.legend();
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.1f'%h,
             ha='center', va='bottom')

    autolabel(rec1)
    autolabel(rec2)

    ymin, ymax = plt.ylim()
    plt.ylim((ymin, ymax+5))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(fr, tr, class_name, save_path):
    roc_auc = sklearn.metrics.auc(fr, tr)
    plt.figure()
    lw = 2
    plt.plot(fr, tr, color='darkorange',
             lw=lw, label='{} ROC curve (area = {:.2f})'.format(class_name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False {} Rate'.format(class_name))
    plt.ylabel('True {} Rate'.format(class_name))
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
