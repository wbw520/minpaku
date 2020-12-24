import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(6, 6), dpi=80)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
linewidth = 2.5
font = 15
markers = ['o', 's', '']
colors = ['#edb03d', "#4dbeeb", "#77ac41"]
epoch = np.arange(20)

p_loss_train = [0.064, 0.05, 0.048, 0.047, 0.046, 0.046, 0.045, 0.045, 0.044, 0.043, 0.043, 0.042, 0.04, 0.037, 0.036, 0.035, 0.035, 0.035, 0.034, 0.034]
p_loss_val = [0.062, 0.061, 0.059, 0.06, 0.058, 0.058, 0.058, 0.058, 0.057, 0.057, 0.057, 0.056, 0.054, 0.053, 0.052, 0.052, 0.052, 0.052, 0.052, 0.052]
train_acc1 = [0.577, 0.682, 0.742, 0.792, 0.832, 0.865, 0.888, 0.907, 0.921, 0.931, 0.939, 0.946, 0.987, 0.997, 0.998, 0.999, 0.999, 0.999, 0.999, 0.999]
val_acc1 = [0.617, 0.644, 0.667, 0.669, 0.68, 0.681, 0.683, 0.683, 0.683, 0.683, 0.686, 0.684, 0.712, 0.715, 0.716, 0.715, 0.712, 0.714, 0.714, 0.713]
ax1.set_title("p loss", fontsize=font+1)
ax2.set_title("acc 1", fontsize=font+1)


ax1.plot(epoch, p_loss_train, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="trian_p")
ax1.plot(epoch, p_loss_val, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="val_p")
ax1.legend(loc='upper right', fontsize=font-4.5, ncol=1)
ax2.plot(epoch, train_acc1, marker=markers[0], markevery=1, markersize=7, color=colors[1], linewidth=linewidth, linestyle="-", label="trian_acc")
ax2.plot(epoch, val_acc1, marker=markers[1], markevery=1, markersize=7, color=colors[2], linewidth=linewidth, linestyle="-", label="val_acc")
ax2.legend(loc='upper right', fontsize=font-4.5, ncol=1)

plt.tight_layout()
# plt.savefig("ablation.pdf")
plt.show()