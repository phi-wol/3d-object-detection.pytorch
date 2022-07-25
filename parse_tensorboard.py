
#%%
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
from scipy import stats
import tensorboard as tb

#%%
major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, \
    "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

# %%
# experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scalars()
# df

# tb.data.experimental.Exp


#%%
!ls

# %%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

path_names = [
    "./tensorboard_samples/tf_logs_1/events.out.tfevents.1651417208.58a4033e3dcf.47407.0",
    "./tensorboard_samples/tf_logs_2/events.out.tfevents.1651602212.58a4033e3dcf.94014.0", 
    "./tensorboard_samples/tf_logs_3/events.out.tfevents.1651662272.58a4033e3dcf.36589.0"
]

def get_event_frame(path_name):
    acc = EventAccumulator(path_name)
    acc.Reload()

    # extract scalar values
    scalars = []
    for tag in acc.Tags()['scalars']:
        pdf = pd.DataFrame(acc.Scalars(tag))
        pdf = pdf.drop('wall_time',axis=1)
        pdf = pdf.rename(columns={'value': tag})
        pdf = pdf.set_index('step')
        scalars.append(pdf)

    # merge single value data frames 
    pd_merged = scalars[0]
    for i in range(1, len(scalars)):
        pd_merged = pd.merge(pd_merged, scalars[i], 'outer', 'step')

    return pd_merged

event_frames = []
for pn in path_names:
    event_frames.append(get_event_frame(pn))

#%%
pd_merged = pd.concat(event_frames, axis=0)

#%% Train Loss
pd_merged['train/loss'].plot()

#%% Train Loss Split
loss_4 = [
    #'train/loss_cls',
    'train/loss_bbox_dims',
    'train/loss_bbox_locs',
    'train/loss_bbox_oris',
]

pd_merged[loss_4 ].plot()


#%% Plot trainings loss functions
plt.figure(figsize=(6,4))
ax = pd_merged['train/loss_cls'].plot()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
plt.xlabel('iterations')
plt.ylabel('classification loss')
plt.tight_layout()
#plt.show()
plt.savefig('loss_cls.pdf')

#%%
plt.figure(figsize=(6,4))
ax = pd_merged['train/loss_bbox_dims'].plot()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
plt.xlabel('iterations')
plt.ylabel('dimension loss')
plt.tight_layout()
#plt.show()
plt.savefig('loss_dims.pdf')

#%%
plt.figure(figsize=(6,4))
ax = pd_merged['train/loss_bbox_locs'].plot()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
plt.xlabel('iterations')
plt.ylabel('translation loss')
plt.tight_layout()
#plt.show()
plt.savefig('loss_locs.pdf')

#%% Plot trainings loss functions
plt.figure(figsize=(6,4))
ax = pd_merged['train/loss_bbox_oris'].plot()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
plt.xlabel('iterations')
plt.ylabel('rotation loss')
plt.tight_layout()
#plt.show()
plt.savefig('loss_oris.pdf')

#%%
'val/0_Mean_Error_2D',
    'val/0_Mean_3D_IoU',
    'val/0_Mean Azimuth Error',
    'val/0_Mean Polar Error',

#%%
plt.figure(figsize=(6,4))
df_2d = pd_merged['val/0_Mean_Error_2D'].copy().dropna()

#df_2d[6730] = 0.149
#df_2d[13460] = 0.118
df_tmp = pd.Series(data=[0.109, 0.098], index=[6730, 13460])
df_2d = pd.concat([df_tmp, df_2d])

ax = df_2d.plot()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
plt.xlabel('iterations')
plt.ylabel('mean error 2D')
plt.tight_layout()
#plt.show()
plt.savefig('val_mean_2d.pdf')

#%%
plt.figure(figsize=(6,4))
df_3d = pd_merged['val/0_Mean_3D_IoU'].copy().dropna()
# ax.set_xlabel("x label")
# ax.set_ylabel("y label")

df_tmp = pd.Series(data=[0.545, 0.565], index=[6730, 13460])
df_3d = pd.concat([df_tmp, df_3d])

ax = df_3d.plot()
plt.xlabel('iterations')
plt.ylabel('mean 3D IoU')
plt.tight_layout()
#plt.show()
plt.savefig('val_mean_3DIoU.pdf')

#%%
plt.figure(figsize=(6,4))
df_az = pd_merged['val/0_Mean Azimuth Error'].copy().dropna()

df_tmp = pd.Series(data=[14.73, 14.545], index=[6730, 13460])
df_az = pd.concat([df_tmp, df_az])

# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
ax = df_az.plot()
plt.xlabel('iterations')
plt.ylabel('mean azimuth error')
plt.tight_layout()
#plt.show()
plt.savefig('val_mean_azimuth.pdf')

#%%
plt.figure(figsize=(6,4))
df_po = pd_merged['val/0_Mean Polar Error'].copy().dropna()

df_tmp = pd.Series(data=[7.09, 6.34], index=[6730, 13460])
df_po = pd.concat([df_tmp, df_po])

# ax.set_xlabel("x label")
# ax.set_ylabel("y label")
ax = df_po.plot()
plt.xlabel('iterations')
plt.ylabel('mean polar error')
plt.tight_layout()
#plt.show()
plt.savefig('val_mean_polar.pdf')


#%% Validation:
loss_val = [
    'val/loss_cls',
    'val/loss_bbox',
    'val/loss_bbox_dims',
    'val/loss_bbox_locs',
    'val/loss_bbox_oris',
    'val/loss']

pd_merged['val/loss'].plot()
pd_merged[loss_val[1:5]].plot()

#%% Metrics:
classes = ['chair', 'book']
classes_dict = {
    '0' : 'chair',
    '1' : 'book'
}
metrics = [
    'val/0_Mean_Error_2D',
    'val/0_Mean_3D_IoU',
    'val/0_Mean Azimuth Error',
    'val/0_Mean Polar Error',
    'val/1_Mean_Error_2D',
    'val/1_Mean_3D_IoU',
    'val/1_Mean Azimuth Error',
    'val/1_Mean Polar Error'
    ]
# for c in classes: 
#     metrics = [
#         f'val/{c}_Mean_Error_2D',
#         f'val/{c}_Mean_3D_IoU',
#         f'val/{c}_Mean Azimuth Error',
#         f'val/{c}_Mean Polar Error'
#     ]
for m in metrics:
    pd_merged[m].plot().set_title(m) #f"metrics {classes_dict[m[4]]}")
    plt.show()

#%%
plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
sns.lineplot(
    data=pd_merged, 
    x="step", 
    y="train/loss")
    #hue=optimizer_validation).set_title("accuracy")

# %% Loss Analysis

loss_4 = [
    'train/loss_cls',
    'train/loss_bbox_dims',
    'train/loss_bbox_locs',
    'train/loss_bbox_oris',
]
    

plt.figure(figsize=(16, 16))
plt.subplot(2, 2, 1)
sns.lineplot(
    data=dfw_validation, 
    x="step", 
    y="epoch_accuracy",
    hue=optimizer_validation).set_title("accuracy")
plt.subplot(2, 2, 2)
sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
             hue=optimizer_validation).set_title("loss")
plt.subplot(2, 2, 3)
sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
             hue=optimizer_validation).set_title("loss")
plt.subplot(2, 2, 4)
sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
             hue=optimizer_validation).set_title("loss")