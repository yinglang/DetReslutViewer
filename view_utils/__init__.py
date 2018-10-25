from .visualize import *
from .dataset import *
from enum import Enum


class VisState(Enum):
    """
        state:
        ------
        0. show current index image, wait key event['left', 'right', 'escape'(->state2), 'enter'(->state1)]
        1. automatic, auto show image one by one, wait key event
                                        ['left'(->state0), 'right'(->state0), 'escape', 'enter'(->state0)]
        2. exit, close all graph.
    """
    normal = 0
    auto = 1
    exit = 2


class LastNextButtonV2(object):
    def __init__(self, fig, axes, args, interval=None):
        # plt.text(-1000, 0., "last", size=50, ha="center", va="center",
        #          bbox=dict(boxstyle="round",
        #                    ec=(1., 0.5, 0.5),
        #                    fc=(1., 0.8, 0.8),
        #                    )
        #          )
        #
        # plt.text(800, 0., "next", size=50, ha="center", va="center",
        #          bbox=dict(boxstyle="round",
        #                    ec=(1., 0.5, 0.5),
        #                    fc=(1., 0.8, 0.8),
        #                    )
        #          )
        self.fig = fig
        self.axes =axes
        fig.canvas.mpl_connect('key_release_event', self.on_key)
        self.args = args
        self.state = VisState.normal
        # self.interval = interval
        self.interval = 0.1 if interval is None else interval

    def on_key(self, event):
        if event.key == 'left':
            self.args['index'] -= 1
            for ax in self.axes.reshape((-1,)):
                ax.cla()
            if self.state == VisState.auto: self.state = VisState.normal
        elif event.key == 'right':
            self.args['index'] += 1
            for ax in self.axes.reshape((-1,)):
                ax.cla()
            if self.state == VisState.auto: self.state = VisState.normal
        elif event.key == 'enter':
            if self.state == VisState.auto: self.state = VisState.normal
            elif self.state == VisState.normal: self.state = VisState.auto
        elif event.key == 'escape':
            self.state = VisState.exit
            self.args['exit'] = True

    def update(self):
        if self.state == VisState.exit:
            plt.close()
            return

        plt.draw()
        plt.pause(self.interval)
        if self.state == VisState.auto:
            self.args['index'] += 1
            for ax in self.axes.reshape((-1,)):
                ax.cla()


class LastNextButtonV1(object):
    def __init__(self, fig, axes, args):
        # plt.text(-1000, 0., "last", size=50, ha="center", va="center",
        #          bbox=dict(boxstyle="round",
        #                    ec=(1., 0.5, 0.5),
        #                    fc=(1., 0.8, 0.8),
        #                    )
        #          )
        #
        # plt.text(800, 0., "next", size=50, ha="center", va="center",
        #          bbox=dict(boxstyle="round",
        #                    ec=(1., 0.5, 0.5),
        #                    fc=(1., 0.8, 0.8),
        #                    )
        #          )
        self.fig = fig
        fig.canvas.mpl_connect('key_release_event', self.on_key)
        self.args = args

    def on_key(self, event):
        if event.key == 'left':
            self.args['index'] -= 1
            plt.close()
        elif event.key == 'right':
            self.args['index'] += 1
            plt.close()
        elif event.key == 'escape':
            self.args['exit'] = True
            plt.close()

    def update(self):
        plt.show()
