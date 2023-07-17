def resample_patchemb(old, new_hw):
  """Resample the weights of the patch embedding kernel to target resolution.

  We resample the patch embedding kernel by approximately inverting the effect
  of patch resizing. Colab with detailed explanation:
  (internal link)
  With this resizing, we can for example load a B/8 filter into a B/16 model
  and, on 2x larger input image, the result will match.
  See (internal link)
  Args:
    old: original parameter to be resized.
    new_hw: target shape (height, width)-only.
  Returns:
    Resized patch embedding kernel.
  """
  assert len(old.shape) == 4, "Four dimensions expected"
  assert len(new_hw) == 2, "New shape should only be hw"
  if tuple(old.shape[:2]) == tuple(new_hw):
    return old

  logging.info("FlexiViT: resize embedding %s to %s", old.shape, new_hw)

  def resize(x_np, new_shape):
    x_tf = tf.constant(x_np)[None, ..., None]
    # NOTE: we are using tf.image.resize here to match the resize operations in
    # the data preprocessing pipeline.
    x_upsampled = tf.image.resize(
        x_tf, new_shape, method="bilinear")[0, ..., 0].numpy()
    return x_upsampled

  def get_resize_mat(old_shape, new_shape):
    mat = []
    for i in range(np.prod(old_shape)):
      basis_vec = np.zeros(old_shape)
      basis_vec[np.unravel_index(i, old_shape)] = 1.
      mat.append(resize(basis_vec, new_shape).reshape(-1))
    return np.stack(mat).T

  resize_mat = get_resize_mat(old.shape[:2], new_hw)
  resize_mat_pinv = np.linalg.pinv(resize_mat.T)

  def resample_kernel(kernel):
    resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
    return resampled_kernel.reshape(new_hw)
  v_resample_kernel = jax.vmap(jax.vmap(resample_kernel, 2, 2), 3, 3)
  return v_resample_kernel(old)