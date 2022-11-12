import numpy as np
import kim

# `tests/test_nd_backend.py` gradient check use `assert error < 4.2e-1`
def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    if kim.array_api == np: tol = 1e-6 # numpy_backend
    else: tol = 4.2e-1 # ndarray_backend
    eps = 1e-4 # = 1^(-4)
    # Khởi tạo mảng numerical_grads = [0..] có shapes tương ứng với 
    # từng args đầu vào của hàm `f`
    numerical_grads = [np.zeros(a.shape) for a in args]

    for i in range(len(args)): # Với từng arg đầu vào của f
        # Với từng phần tử trong numpy array `args[i].realize_cached_data()`
        for j in range(args[i].realize_cached_data().size):
            # Cộng phần tử thứ j của args[i] thêm epsilon
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            # Trừ phần tử thứ j của args[i] đi epsilon
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            # Trả lại giá trị nguyên bản của phần tử thứ j của args[i]
            args[i].realize_cached_data().flat[j] += eps
            # Tính numerical_grad của phần tử thứ j của args[i]
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    
    if not backward: # `backward` mặc định là False => Sử dụng hàm `gradient()`
        # kim.broadcast_to và kim.summation rơi vào trường hợp này
        out_node = f(*args, **kwargs)
        out_grad = kim.Tensor(np.ones(out_node.shape))
        computed_grads = [ x.numpy()
            for x in out_node.op.gradient(out_grad, out_node) ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]

    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )

    # print(">>>", f, args, kwargs)
    print(">>>", numerical_grads)
    print(">>>", computed_grads)
    assert error < tol
    return computed_grads
