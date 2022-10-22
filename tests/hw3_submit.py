from test_ndarray import *

######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:128], A.strides, A.shape)


def Rand(*shape, device=nd.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=100, size=shape)
    return nd.array(_A, device=device)


def RandC(*shape, entropy=1):
    if nd.cuda().enabled():
        return Rand(*shape, device=nd.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")


def MugradeSubmit(things):
    mugrade.submit(Prepare(things))


def submit_ndarray_python_ops():
    MugradeSubmit(Rand(4, 4).reshape((2, 2, 4)))
    MugradeSubmit(Rand(2, 2, 4).reshape((4, 4)))
    MugradeSubmit(Rand(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)))
    MugradeSubmit(Rand(2, 1, 2).broadcast_to((2, 3, 2)))
    MugradeSubmit(Rand(1, 1, 2).broadcast_to((2, 2, 2)))
    MugradeSubmit(Rand(4, 4)[1, 0])
    MugradeSubmit(Rand(4, 4)[1, 0:3])
    MugradeSubmit(Rand(4, 4)[1:3, 0:3])
    MugradeSubmit(Rand(4, 4, 4)[0:2, 1:3, 2:4])
    MugradeSubmit(Rand(4, 4, 4)[0:4:2, :3, 2:])


def submit_ndarray_cpu_compact_setitem():
    MugradeSubmit(Rand(4, 4, 4)[0:4:2, :3, 2:].compact())
    MugradeSubmit(Rand(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)).compact())
    MugradeSubmit(Rand(1, 1, 2).broadcast_to((2, 2, 2)).compact())
    MugradeSubmit(Rand(4, 4).reshape((2, 2, 4)).compact())

    A = Rand(4, 4)
    B = Rand(4, 4)
    A[1, 0:3] = B[0, 1:4]
    MugradeSubmit(A)

    A = Rand(4, 4)
    B = Rand(4, 4)
    A[0:3, 1] = B[1:4, 0]
    MugradeSubmit(A)

    A = Rand(2, 2, 2, 3)
    B = Rand(2, 2, 2, 3)
    A[0, :, 1, :2] = B[0, :, 1, :2]
    MugradeSubmit(A)

    A = Rand(2, 2, 2, 3)
    A[0, :, 1, :2] = 42.0
    MugradeSubmit(A)


def submit_ndarray_cpu_ops():
    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A * B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A / B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A == B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A >= B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A.maximum(B))

    A = Rand(2, 2)
    MugradeSubmit(A * 42.0)

    A = Rand(2, 2)
    MugradeSubmit(A ** 2.0)

    A = Rand(2, 2)
    MugradeSubmit(A / 42.0)

    A = Rand(5, 5)
    MugradeSubmit(A.maximum(25.0))

    A = Rand(10, 10)
    MugradeSubmit(A == 10)

    A = Rand(5, 5)
    MugradeSubmit(A > 50)

    A = Rand(2, 2)
    MugradeSubmit(A.log())

    A = Rand(2, 2)
    MugradeSubmit(A.tanh())

    A = Rand(2, 2)
    MugradeSubmit((A/100).exp())


def submit_ndarray_cpu_reductions():
    MugradeSubmit(Rand(4, 4).sum(axis=1))
    MugradeSubmit(Rand(4, 4, 4).sum(axis=1))
    MugradeSubmit(Rand(4).sum(axis=0))
    MugradeSubmit(Rand(2, 2, 2, 4, 2).sum(axis=3))
    MugradeSubmit(Rand(4, 4).max(axis=1))
    MugradeSubmit(Rand(4, 4, 4).max(axis=1))
    MugradeSubmit(Rand(4).max(axis=0))
    MugradeSubmit(Rand(2, 2, 2, 4, 2).max(axis=3))


def submit_ndarray_cpu_matmul():
    A, B = Rand(4, 4), Rand(4, 4)
    MugradeSubmit(A @ B)

    A, B = Rand(3, 4), Rand(4, 3)
    MugradeSubmit(A @ B)

    A, B = Rand(73, 72), Rand(72, 73)
    MugradeSubmit(A @ B)

    # tiled
    for m, n, p in [(3, 2, 1), (3, 3, 3), (3, 4, 5)]:
        device = nd.cpu()
        t = device.__tile_size__
        A = Rand(m, n, t, t)
        B = Rand(n, p, t, t)
        C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
        device.matmul_tiled(A._handle, B._handle, C._handle, m*t, n*t, p*t)
        MugradeSubmit(C)


def submit_ndarray_cuda_compact_setitem():
    MugradeSubmit(RandC(4, 4, 4)[0:4:2, :3, 2:].compact())
    MugradeSubmit(RandC(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)).compact())
    MugradeSubmit(RandC(1, 1, 2).broadcast_to((2, 2, 2)).compact())
    MugradeSubmit(RandC(4, 4).reshape((2, 2, 4)).compact())

    A = RandC(4, 4)
    B = RandC(4, 4)
    A[1, 0:3] = B[0, 1:4]
    MugradeSubmit(A)

    A = RandC(4, 4)
    B = RandC(4, 4)
    A[0:3, 1] = B[1:4, 0]
    MugradeSubmit(A)

    A = RandC(2, 2, 2, 3)
    B = RandC(2, 2, 2, 3)
    A[0, :, 1, :2] = B[0, :, 1, :2]
    MugradeSubmit(A)

    A = RandC(2, 2, 2, 3)
    A[0, :, 1, :2] = 42.0
    MugradeSubmit(A)


def submit_ndarray_cuda_ops():
    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A * B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A / B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A == B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A >= B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A.maximum(B))

    A = RandC(2, 2)
    MugradeSubmit(A * 42.0)

    A = RandC(2, 2)
    MugradeSubmit(A ** 2.0)

    A = RandC(2, 2)
    MugradeSubmit(A / 42.0)

    A = RandC(5, 5)
    MugradeSubmit(A.maximum(25.0))

    A = RandC(10, 10)
    MugradeSubmit(A == 10)

    A = RandC(5, 5)
    MugradeSubmit(A > 50)

    A = RandC(2, 2)
    MugradeSubmit(A.log())

    A = RandC(2, 2)
    MugradeSubmit(A.tanh())

    A = RandC(2, 2)
    MugradeSubmit((A/100).exp())


def submit_ndarray_cuda_reductions():
    MugradeSubmit(RandC(4, 4).sum(axis=1))
    MugradeSubmit(RandC(4, 4, 4).sum(axis=1))
    MugradeSubmit(RandC(4).sum(axis=0))
    MugradeSubmit(RandC(2, 2, 2, 4, 2).sum(axis=3))
    MugradeSubmit(RandC(4, 4).max(axis=1))
    MugradeSubmit(RandC(4, 4, 4).max(axis=1))
    MugradeSubmit(RandC(4).max(axis=0))
    MugradeSubmit(RandC(2, 2, 2, 4, 2).max(axis=3))


def submit_ndarray_cuda_matmul():
    A, B = RandC(4, 4), RandC(4, 4)
    MugradeSubmit(A @ B)

    A, B = RandC(4, 3), RandC(3, 4)
    MugradeSubmit(A @ B)

    A, B = RandC(3, 4), RandC(4, 3)
    MugradeSubmit(A @ B)

    A, B = RandC(33, 33), RandC(33, 33)
    MugradeSubmit(A @ B)

    A, B = RandC(73, 72), RandC(72, 71)
    MugradeSubmit(A @ B)

    A, B = RandC(123, 125), RandC(125, 129)
    MugradeSubmit(A @ B)
