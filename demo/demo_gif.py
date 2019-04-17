# -*- coding: utf-8 -*-
"""
@ Description: 
-------------

-------------
@ Time    : 2018/11/25 21:21
@ Author  : Yaoming Cai
@ FileName: demo_gif.py
@ Software: PyCharm
@ Blog    ：https://github.com/AngryCai
@ Email   : caiyaomxc@outlook.com
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
fig.set_tight_layout(True)


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# path = 'C:\\Users\\07\Desktop\putty\history.npz'
path = 'F:\Python\DeepLearning\AttentionBandSelection\\results\history-PaviaU-KNN-SVM-multiscale-5band.npz'
npz = np.load(path)
# s = npz['score']
s1 = npz['score'][:, 0, 0, 0]
s2 = npz['score'][:, 1, 0, 0]
loss_ = npz['loss']
w = npz['channel_weight']

im = ax.imshow(w[0][:100], animated=True)


def update(i):
    label = 'Epoch {0}'.format(i)
    print(label)
    im.set_data(w[i][:100])
    ax.set_xlabel(label)
    return im


anim = animation.FuncAnimation(fig, update, frames=100, interval=400)

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

# To save the animation, use e.g.
#
# anim.save("C:\\Users\\07\Desktop\Indian_animation.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
# anim.save('C:\\Users\\07\Desktop\Indian_animation.gif', dpi=100, writer='ffmpeg')

