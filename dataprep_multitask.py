import os
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_h5_file_multitask(file_path):
    with h5py.File(file_path, 'r') as f:
        image = f['image'][()]  # (240, 240, 4)
        mask = f['mask'][()]    # (240, 240) OR (240, 240, 3)

    # Normalize image
    image = (image - np.mean(image, axis=(0, 1), keepdims=True)) / \
            (np.std(image, axis=(0, 1), keepdims=True) + 1e-6)

    # Handle case if mask is already one-hot encoded (3-channel)
    if mask.ndim == 3 and mask.shape[-1] == 3:
        # One-hot format (NCR, ED, ET) → derive binary targets
        ncr = mask[..., 0]
        ed = mask[..., 1]
        et = mask[..., 2]

        wt = ((ncr + ed + et) > 0).astype(np.float32)[..., np.newaxis]  # Whole tumor
        tc = ((ncr + et) > 0).astype(np.float32)[..., np.newaxis]       # Tumor core
        et = (et > 0).astype(np.float32)[..., np.newaxis]               # Enhancing tumor
    else:
        # Single-channel label map
        mask = mask.astype(np.uint8)
        wt = (mask > 0).astype(np.float32)[..., np.newaxis]
        tc = np.isin(mask, [1, 4]).astype(np.float32)[..., np.newaxis]
        et = (mask == 4).astype(np.float32)[..., np.newaxis]

    return image.astype(np.float32), wt, tc, et


def _parse_multitask_function(path):
    image, wt, tc, et = tf.numpy_function(
        load_h5_file_multitask, [path],
        [tf.float32, tf.float32, tf.float32, tf.float32]
    )

    image.set_shape((240, 240, 4))
    wt.set_shape((240, 240, 1))
    tc.set_shape((240, 240, 1))
    et.set_shape((240, 240, 1))

    masks = {
        'wt_head': wt,
        'tc_head': tc,
        'et_head': et
    }

    return image, masks


def get_dataset(file_paths, batch_size=8, shuffle=False, num_workers=4):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(_parse_multitask_function, num_parallel_calls=num_workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_train_val_datasets(data_dir, batch_size=8, test_size=0.2, random_state=42):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=random_state)

    train_dataset = get_dataset(train_files, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = get_dataset(val_files, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    return train_dataset, val_dataset
