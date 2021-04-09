data = dict(
    root="./data",
    resize=(224,224),
    train_batch_size=64,
    val_batch_size=32,
    max_epochs=200,
    num_workers=4,
    category_list='all',
    normalization=dict(mean=[0.5931, 0.4690, 0.4229],
                       std=[0.2471, 0.2214, 0.2157])
)

model = dict(name='mobilenetv3_large', pretrained=True, num_classes=1, load_weights='')

data_parallel = dict(use_parallel=True,
                     parallel_params=dict(device_ids=[0], output_device=0))

optim = dict(name='adam', lr=0.001, momentum=0.9, wd=1e-4, betas=(0.9, 0.999), rho=0.9, alpha=0.99)

scheduler = dict(name='exp', gamma=0.1, exp_gamma=0.975, steps=[50])

loss=dict(names=['mse', 'add_loss'], coeffs=([1., .1],[]), smoothl1_beta=0.2)

utils=dict(debug_mode = False, random_seeds=5, save_freq=10, print_freq=20, debug_steps=30)

output_dir = './output/exp_0'

regime = dict(type='training', vis_only=False)
